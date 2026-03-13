# IMP: Elasticsearch 기반 하이브리드 검색 도구 모듈.
# BM25 (키워드 검색) + kNN (벡터 유사도 검색)을 수동 RRF로 결합하여
# 정확도와 의미적 유사성을 모두 활용하는 하이브리드 검색을 구현합니다.

from langchain_core.tools import tool
from app.core.config import settings
from datetime import datetime, timedelta

import requests
import xmltodict
from elasticsearch import Elasticsearch
from openai import OpenAI
import warnings

warnings.filterwarnings("ignore")

# ─── 클라이언트 초기화 ──────────────────────────────────────────────────────
KOPIS_BASE_URL = "http://www.kopis.or.kr/openApi/restful/"
INDEX_NAME = "performances"
EMBEDDING_MODEL = "text-embedding-3-small"


def _get_es_client() -> Elasticsearch:
    """ES 클라이언트 (싱글톤 패턴)"""
    if not hasattr(_get_es_client, "_client"):
        # SEC: ⚠️ SECURITY WARNING — 프로덕션 환경에서는 반드시 verify_certs=True로 변경하세요.
        # verify_certs=False는 SSL 인증서 검증을 비활성화하여 MITM(중간자 공격)에 취약합니다.
        _get_es_client._client = Elasticsearch(
            settings.ES_URL,
            basic_auth=(settings.ES_USERNAME, settings.ES_PASSWORD),
            verify_certs=False,
            request_timeout=30,
        )
    return _get_es_client._client


def _get_openai_client() -> OpenAI:
    """OpenAI 클라이언트 (싱글톤 패턴)"""
    if not hasattr(_get_openai_client, "_client"):
        _get_openai_client._client = OpenAI(api_key=settings.OPENAI_API_KEY)
    return _get_openai_client._client


# ─── 지역/장르 코드 매핑 ────────────────────────────────────────────────────
REGION_CODE_MAP = {
    "서울": "11", "부산": "26", "대구": "27", "인천": "28",
    "광주": "29", "대전": "30", "울산": "31", "세종": "36",
    "경기": "41", "강원": "42", "충북": "43", "충남": "44",
    "전북": "45", "전남": "46", "경북": "47", "경남": "48", "제주": "50",
}

GENRE_CODE_MAP = {
    "연극": "AAAA", "뮤지컬": "GGGA", "무용": "BBBC",
    "클래식": "CCCA", "오페라": "CCCC", "국악": "CCCD", "복합": "EEEA",
}

# KOPIS genrenm → 우리 장르명 매핑 (ES에 저장된 값과 일치시키기 위해)
KOPIS_GENRE_MAP = {
    "연극": "연극",
    "뮤지컬": "뮤지컬",
    "무용(서양/한국무용)": "무용",
    "서양음악(클래식)": "클래식",
    "한국음악(국악)": "국악",
    "대중음악": "대중음악",
    "복합": "복합",
}


# ─── 임베딩 생성 ──────────────────────────────────────────────────────────────
def _generate_query_embedding(query: str) -> list[float]:
    """검색 쿼리의 벡터 임베딩을 생성합니다."""
    client = _get_openai_client()
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[query],
    )
    return response.data[0].embedding


# ─── 수동 RRF (Reciprocal Rank Fusion) ───────────────────────────────────────
# IMP: ES Basic 라이선스에서는 내장 RRF를 사용할 수 없으므로,
# BM25와 kNN 결과를 각각 가져온 뒤 Python에서 RRF 점수를 계산하여 결합합니다.
# RRF 공식: score(d) = Σ 1/(k + rank_i(d))  (k=60이 기본값)
def _manual_rrf(results_list: list[dict], k: int = 60, top_n: int = 10) -> list[dict]:
    """여러 검색 결과를 RRF로 결합하여 최종 랭킹을 산출합니다."""
    scores = {}
    docs = {}

    for results in results_list:
        hits = results.get("hits", {}).get("hits", [])
        for rank, hit in enumerate(hits):
            doc_id = hit["_id"]
            rrf_score = 1.0 / (k + rank + 1)
            scores[doc_id] = scores.get(doc_id, 0) + rrf_score
            if doc_id not in docs:
                docs[doc_id] = hit["_source"]

    sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [docs[doc_id] for doc_id in sorted_ids[:top_n]]


# ─── ES 검색 헬퍼 ─────────────────────────────────────────────────────────────
def _build_filters(genre: str = "", region: str = "",
                   start_date: str = "", end_date: str = "") -> list[dict]:
    """ES 쿼리용 필터 조건을 생성합니다."""
    filters = []
    if genre:
        # ES에 저장된 장르명으로 매핑
        es_genre = None
        for kopis_genre, our_genre in KOPIS_GENRE_MAP.items():
            if our_genre == genre or kopis_genre == genre:
                es_genre = kopis_genre
                break
        if es_genre:
            filters.append({"term": {"genre": es_genre}})
    if region and region in REGION_CODE_MAP:
        filters.append({"term": {"region": region}})
    if start_date:
        filters.append({"range": {"end_date": {"gte": start_date}}})
    if end_date:
        filters.append({"range": {"start_date": {"lte": end_date}}})
    return filters


def _hybrid_search(query: str, filters: list[dict] = None,
                   size: int = 20, top_n: int = 10) -> list[dict]:
    """BM25 + kNN 하이브리드 검색을 수행하고 RRF로 결합합니다."""
    es = _get_es_client()

    if filters is None:
        filters = []

    # 쿼리 임베딩 생성
    query_vector = _generate_query_embedding(query)

    # 1) BM25 검색 (텍스트 기반 키워드 매칭)
    bm25_body = {
        "size": size,
        "_source": {"excludes": ["embedding"]},
        "query": {
            "bool": {
                "should": [
                    {"multi_match": {
                        "query": query,
                        "fields": ["name^3", "combined_text^2", "venue", "cast"],
                    }},
                ],
                "filter": filters,
                "minimum_should_match": 0,
            }
        },
    }

    # 필터만 있고 should가 매칭되지 않아도 결과를 반환하도록 설정
    if filters:
        bm25_body["query"]["bool"]["should"].append({"match_all": {}})

    bm25_result = es.search(index=INDEX_NAME, body=bm25_body)

    # 2) kNN 검색 (벡터 유사도 기반 시맨틱 매칭)
    knn_body = {
        "size": size,
        "_source": {"excludes": ["embedding"]},
        "knn": {
            "field": "embedding",
            "query_vector": query_vector,
            "k": size,
            "num_candidates": size * 5,
        },
    }
    if filters:
        knn_body["knn"]["filter"] = {"bool": {"filter": filters}}

    knn_result = es.search(index=INDEX_NAME, body=knn_body)

    # 3) RRF 결합
    combined = _manual_rrf([bm25_result, knn_result], top_n=top_n)
    return combined


# ─── 포맷팅 헬퍼 ──────────────────────────────────────────────────────────────
def _format_performance(doc: dict) -> str:
    """ES 문서를 읽기 좋은 문자열로 변환"""
    name = doc.get("name", "알 수 없음")
    venue = doc.get("venue", "알 수 없음")
    start = doc.get("start_date", "")
    end = doc.get("end_date", "")
    genre = doc.get("genre", "")
    state = doc.get("state", "")
    region = doc.get("region", "")
    perf_id = doc.get("performance_id", "")
    return f"[{perf_id}] {name} | {genre} | {region} | {venue} | {start}~{end} | {state}"


# ─── LangChain 도구 정의 ─────────────────────────────────────────────────────
@tool
def search_performances(
    keyword: str = "",
    genre: str = "",
    region: str = "",
    start_date: str = "",
    end_date: str = "",
) -> str:
    """공연/전시회를 검색합니다. 키워드, 장르, 지역, 기간으로 필터링할 수 있습니다.
    Elasticsearch 하이브리드 검색(BM25 + kNN + RRF)을 사용하여 정확도 높은 결과를 반환합니다.

    Args:
        keyword: 공연 이름 키워드 (예: "오페라의 유령", "레미제라블")
        genre: 장르 (연극, 뮤지컬, 무용, 클래식, 오페라, 국악, 복합)
        region: 지역 (서울, 부산, 대구, 인천, 광주, 대전, 울산, 세종, 경기, 강원, 충북, 충남, 전북, 전남, 경북, 경남, 제주)
        start_date: 검색 시작 날짜 (YYYYMMDD 형식, 기본값: 오늘)
        end_date: 검색 종료 날짜 (YYYYMMDD 형식, 기본값: 90일 후)

    Returns:
        검색된 공연 목록 정보
    """
    try:
        if not start_date:
            start_date = datetime.now().strftime("%Y%m%d")
        if not end_date:
            end_date = (datetime.now() + timedelta(days=90)).strftime("%Y%m%d")

        # 필터 조건 구성
        filters = _build_filters(genre, region, start_date, end_date)

        # 검색 쿼리 구성 (키워드가 없으면 필터 + 장르/지역 정보로 검색)
        search_query = keyword if keyword else f"{region} {genre} 공연".strip()

        # IMP: 하이브리드 검색 실행 (BM25 텍스트 검색 + kNN 벡터 검색 + RRF 결합)
        results = _hybrid_search(search_query, filters=filters, top_n=10)

        if not results:
            return "검색 결과가 없습니다. 다른 조건으로 다시 검색해보세요."

        output_lines = [f"총 {len(results)}건의 공연이 검색되었습니다.\n"]
        for doc in results:
            output_lines.append(f"- {_format_performance(doc)}")

        return "\n".join(output_lines)
    except Exception as e:
        return f"공연 검색 중 오류가 발생했습니다: {str(e)}"


@tool
def get_performance_detail(performance_id: str) -> str:
    """특정 공연의 상세 정보를 조회합니다. search_performances 결과에서 얻은 공연 ID가 필요합니다.
    ES에서 기본 정보를 조회하고, KOPIS API에서 최신 상세 정보를 보완합니다.

    Args:
        performance_id: 공연 ID (예: PF12345). search_performances 결과의 대괄호 안에 있는 ID입니다.

    Returns:
        공연 상세 정보 (제목, 기간, 장소, 가격, 출연진 등)
    """
    try:
        es = _get_es_client()

        # 1) ES에서 기본 정보 조회
        try:
            es_doc = es.get(index=INDEX_NAME, id=performance_id, source_excludes=["embedding"])
            doc = es_doc["_source"]
        except Exception:
            doc = {}

        # 2) KOPIS API에서 최신 상세 정보 보완
        url = f"{KOPIS_BASE_URL}pblprfr/{performance_id}"
        params = {"service": settings.KOPIS_API_KEY}
        # SEC: ⚠️ SECURITY WARNING — 프로덕션 환경에서는 verify=False를 제거하세요.
        response = requests.get(url, params=params, verify=False)
        response.raise_for_status()

        data = xmltodict.parse(response.text)
        detail = data.get("dbs", {}).get("db", {})

        if not detail and not doc:
            return f"공연 ID '{performance_id}'에 대한 정보를 찾을 수 없습니다."

        # ES와 KOPIS 데이터 병합 (KOPIS 우선)
        name = detail.get("prfnm") or doc.get("name", "알 수 없음")
        info_lines = [
            f"공연 상세 정보",
            f"",
            f"공연명: {name}",
            f"기간: {detail.get('prfpdfrom', doc.get('start_date', ''))} ~ {detail.get('prfpdto', doc.get('end_date', ''))}",
            f"장소: {detail.get('fcltynm', doc.get('venue', '알 수 없음'))}",
            f"지역: {doc.get('region', '정보 없음')}",
            f"장르: {detail.get('genrenm', doc.get('genre', ''))}",
            f"출연진: {detail.get('prfcast', doc.get('cast', '정보 없음'))}",
            f"제작진: {detail.get('prfcrew', doc.get('crew', '정보 없음'))}",
            f"런타임: {detail.get('prfruntime', doc.get('runtime', '정보 없음'))}",
            f"관람연령: {detail.get('prfage', doc.get('age', '정보 없음'))}",
            f"가격: {detail.get('pcseguidance', doc.get('price', '정보 없음'))}",
            f"공연시간: {detail.get('dtguidance', doc.get('schedule', '정보 없음'))}",
            f"공연상태: {detail.get('prfstate', doc.get('state', ''))}",
        ]

        return "\n".join(info_lines)
    except Exception as e:
        return f"공연 상세 조회 중 오류가 발생했습니다: {str(e)}"


@tool
def recommend_performances(
    genre: str = "",
    region: str = "서울",
) -> str:
    """현재 공연 중이거나 예정인 작품 중에서 장르와 지역을 기반으로 공연을 추천합니다.
    Elasticsearch 하이브리드 검색을 사용하여 관련도 높은 추천 결과를 제공합니다.

    Args:
        genre: 선호 장르 (연극, 뮤지컬, 무용, 클래식, 오페라, 국악, 복합). 빈 문자열이면 전체 장르.
        region: 선호 지역 (서울, 부산, 대구 등). 기본값: 서울

    Returns:
        추천 공연 목록
    """
    try:
        today = datetime.now().strftime("%Y%m%d")
        week_later = (datetime.now() + timedelta(days=30)).strftime("%Y%m%d")

        # 필터 조건 구성
        filters = _build_filters(genre, region, today, week_later)

        # 추천 쿼리 생성
        search_query = f"{region} {genre} 추천 공연 인기".strip()

        # 하이브리드 검색
        results = _hybrid_search(search_query, filters=filters, top_n=5)

        if not results:
            return "현재 추천할 공연이 없습니다. 다른 장르나 지역으로 시도해보세요."

        genre_label = genre if genre else "전체"
        output_lines = [f"{region} 지역 {genre_label} 장르 추천 공연\n"]
        for doc in results:
            output_lines.append(f"- {_format_performance(doc)}")

        if len(results) >= 5:
            output_lines.append(f"\n상위 5건을 추천합니다.")

        return "\n".join(output_lines)
    except Exception as e:
        return f"공연 추천 중 오류가 발생했습니다: {str(e)}"


@tool
def ChatResponse(message_id: str, content: str, metadata: dict = {}) -> str:
    """최종 응답을 사용자에게 전달하기 위한 도구입니다. 반드시 이 도구를 사용하여 최종 답변을 반환하세요.

    Args:
        message_id: 고유 메시지 ID (UUID 형식으로 생성)
        content: 사용자에게 전달할 최종 답변 내용
        metadata: 추가 메타데이터 (기본값: 빈 딕셔너리)

    Returns:
        응답 전달 확인
    """
    return f"Response delivered: {content[:100]}..."
