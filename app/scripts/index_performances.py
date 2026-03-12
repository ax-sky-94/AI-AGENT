"""
KOPIS 공연 데이터를 Elasticsearch에 인덱싱하는 스크립트.

실행 방법:
    cd /Users/song-kyung-yoon/ai_agent/agent
    .venv/bin/python -m app.scripts.index_performances

주요 기능:
    1. KOPIS API에서 전체 장르별 공연 데이터 수집
    2. 각 공연의 상세 정보(가격, 출연진, 일정 등) 조회
    3. OpenAI text-embedding-3-small 모델로 벡터 임베딩 생성
    4. ES 인덱스 생성 (text + dense_vector 하이브리드 검색 지원)
    5. Bulk 인덱싱으로 효율적 데이터 적재
"""
import os
import time
import warnings
from datetime import datetime, timedelta

import requests
import xmltodict
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from openai import OpenAI

# SSL 경고 억제
warnings.filterwarnings("ignore")

# ─── 설정 ────────────────────────────────────────────────────────────────────
KOPIS_API_KEY = os.getenv("KOPIS_API_KEY")
ES_URL = os.getenv("ES_URL")
ES_USERNAME = os.getenv("ES_USERNAME")
ES_PASSWORD = os.getenv("ES_PASSWORD")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INDEX_NAME = "performances"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMS = 1536

KOPIS_BASE_URL = "http://www.kopis.or.kr/openApi/restful/"

# 수집 대상 장르 (전체)
GENRE_CODE_MAP = {
    "연극": "AAAA",
    "뮤지컬": "GGGA",
    "무용": "BBBC",
    "클래식": "CCCA",
    "오페라": "CCCC",
    "국악": "CCCD",
    "복합": "EEEA",
}

# 수집 대상 지역 (전체)
REGION_CODE_MAP = {
    "서울": "11",
    "부산": "26",
    "대구": "27",
    "인천": "28",
    "광주": "29",
    "대전": "30",
    "울산": "31",
    "세종": "36",
    "경기": "41",
    "강원": "42",
    "충북": "43",
    "충남": "44",
    "전북": "45",
    "전남": "46",
    "경북": "47",
    "경남": "48",
    "제주": "50",
}

# 역방향 매핑 (코드 → 한글)
REGION_NAME_MAP = {v: k for k, v in REGION_CODE_MAP.items()}
GENRE_NAME_MAP = {v: k for k, v in GENRE_CODE_MAP.items()}

# KOPIS area 필드 → 우리 지역명 매핑 (상세 API의 area 필드에서 추출)
AREA_TO_REGION = {
    "서울특별시": "서울", "서울": "서울",
    "부산광역시": "부산", "부산": "부산",
    "대구광역시": "대구", "대구": "대구",
    "인천광역시": "인천", "인천": "인천",
    "광주광역시": "광주", "광주": "광주",
    "대전광역시": "대전", "대전": "대전",
    "울산광역시": "울산", "울산": "울산",
    "세종특별자치시": "세종", "세종": "세종",
    "경기도": "경기", "경기": "경기",
    "강원도": "강원", "강원특별자치도": "강원", "강원": "강원",
    "충청북도": "충북", "충북": "충북",
    "충청남도": "충남", "충남": "충남",
    "전라북도": "전북", "전북특별자치도": "전북", "전북": "전북",
    "전라남도": "전남", "전남": "전남",
    "경상북도": "경북", "경북": "경북",
    "경상남도": "경남", "경남": "경남",
    "제주특별자치도": "제주", "제주도": "제주", "제주": "제주",
}


# ─── ES 인덱스 매핑 정의 ──────────────────────────────────────────────────────
# IMP: BM25 (text 필드) + kNN (dense_vector 필드) 하이브리드 검색을 지원합니다.
# OpenAI text-embedding-3-small 모델 (1536차원)으로 벡터를 생성하여 저장합니다.
INDEX_MAPPING = {
    "settings": {
        "number_of_replicas": 0,
    },
    "mappings": {
        "properties": {
            # 구조화된 필드 (필터링용)
            "performance_id": {"type": "keyword"},
            "genre": {"type": "keyword"},
            "genre_code": {"type": "keyword"},
            "region": {"type": "keyword"},
            "region_code": {"type": "keyword"},
            "state": {"type": "keyword"},
            "start_date": {"type": "date", "format": "yyyyMMdd||yyyy.MM.dd||yyyy-MM-dd"},
            "end_date": {"type": "date", "format": "yyyyMMdd||yyyy.MM.dd||yyyy-MM-dd"},
            # BM25 텍스트 검색용 필드
            "name": {"type": "text", "analyzer": "standard"},
            "venue": {"type": "text", "analyzer": "standard"},
            "cast": {"type": "text", "analyzer": "standard"},
            "crew": {"type": "text", "analyzer": "standard"},
            "price": {"type": "text"},
            "runtime": {"type": "keyword"},
            "age": {"type": "keyword"},
            "schedule": {"type": "text"},
            "poster_url": {"type": "keyword"},
            # BM25 검색을 위한 결합 텍스트 필드
            "combined_text": {"type": "text", "analyzer": "standard"},
            # IMP: dense_vector - OpenAI 임베딩을 저장하는 벡터 필드.
            # kNN 검색 시 cosine similarity로 유사도를 계산합니다.
            "embedding": {
                "type": "dense_vector",
                "dims": EMBEDDING_DIMS,
                "index": True,
                "similarity": "cosine",
            },
        }
    },
}


def get_es_client() -> Elasticsearch:
    """Elasticsearch 클라이언트 생성"""
    return Elasticsearch(
        ES_URL,
        basic_auth=(ES_USERNAME, ES_PASSWORD),
        verify_certs=False,
        request_timeout=60,
    )


def get_openai_client() -> OpenAI:
    """OpenAI 클라이언트 생성"""
    return OpenAI(api_key=OPENAI_API_KEY)


def kopis_request(endpoint: str, params: dict) -> list[dict]:
    """KOPIS API 요청"""
    url = f"{KOPIS_BASE_URL}{endpoint}"
    params["service"] = KOPIS_API_KEY
    response = requests.get(url, params=params, verify=False)
    response.raise_for_status()
    data = xmltodict.parse(response.text)
    db_data = data.get("dbs", {}).get("db")
    if not db_data:
        return []
    if isinstance(db_data, dict):
        return [db_data]
    return db_data


def fetch_performance_list(genre_code: str = "", region_code: str = "",
                           start_date: str = "", end_date: str = "",
                           rows: int = 50, cpage: int = 1) -> list[dict]:
    """공연 목록 조회"""
    if not start_date:
        start_date = datetime.now().strftime("%Y%m%d")
    if not end_date:
        end_date = (datetime.now() + timedelta(days=90)).strftime("%Y%m%d")

    params = {
        "stdate": start_date,
        "eddate": end_date,
        "cpage": str(cpage),
        "rows": str(rows),
    }
    if genre_code:
        params["shcate"] = genre_code
    if region_code:
        params["signgucode"] = region_code

    return kopis_request("pblprfr", params)


def fetch_performance_detail(perf_id: str) -> dict:
    """공연 상세 정보 조회"""
    url = f"{KOPIS_BASE_URL}pblprfr/{perf_id}"
    params = {"service": KOPIS_API_KEY}
    response = requests.get(url, params=params, verify=False)
    response.raise_for_status()
    data = xmltodict.parse(response.text)
    return data.get("dbs", {}).get("db", {})


def build_combined_text(name, genre, venue, region, cast, state, price) -> str:
    """시맨틱 검색용 결합 텍스트 생성"""
    parts = [f"공연명: {name}"]
    if genre:
        parts.append(f"장르: {genre}")
    if venue:
        parts.append(f"장소: {venue}")
    if region:
        parts.append(f"지역: {region}")
    if cast and cast != "정보 없음":
        parts.append(f"출연진: {cast}")
    if state:
        parts.append(f"상태: {state}")
    if price and price != "정보 없음":
        parts.append(f"가격: {price}")
    return " | ".join(parts)


def generate_embeddings(openai_client: OpenAI, texts: list[str], batch_size: int = 100) -> list[list[float]]:
    """OpenAI API로 텍스트 임베딩을 배치로 생성"""
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = openai_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch,
        )
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)
        if i + batch_size < len(texts):
            time.sleep(0.5)  # Rate limit 방지
    return all_embeddings


def build_document(perf: dict, detail: dict) -> dict:
    """KOPIS 데이터를 ES 문서로 변환 (임베딩 제외)"""
    name = detail.get("prfnm") or perf.get("prfnm", "")
    genre = detail.get("genrenm") or perf.get("genrenm", "")
    venue = detail.get("fcltynm") or perf.get("fcltynm", "")
    cast = detail.get("prfcast", "") or ""
    crew = detail.get("prfcrew", "") or ""
    state = detail.get("prfstate") or perf.get("prfstate", "")
    price = detail.get("pcseguidance", "") or ""
    runtime = detail.get("prfruntime", "") or ""
    age = detail.get("prfage", "") or ""
    schedule = detail.get("dtguidance", "") or ""
    poster = detail.get("poster") or perf.get("poster", "")

    start_date = (detail.get("prfpdfrom") or perf.get("prfpdfrom", "")).replace(".", "").replace("-", "")
    end_date = (detail.get("prfpdto") or perf.get("prfpdto", "")).replace(".", "").replace("-", "")

    # IMP: 지역 추출 - 상세 API의 'area' 필드에서 지역명을 매핑합니다.
    area = detail.get("area", "") or ""
    region = AREA_TO_REGION.get(area.strip(), "")
    region_code = REGION_CODE_MAP.get(region, "")

    genre_code = perf.get("shcate", "")

    combined_text = build_combined_text(name, genre, venue, region, cast, state, price)

    return {
        "performance_id": perf.get("mt20id", ""),
        "name": name,
        "genre": genre,
        "genre_code": genre_code,
        "region": region,
        "region_code": region_code,
        "venue": venue,
        "cast": cast,
        "crew": crew,
        "state": state,
        "price": price,
        "runtime": runtime,
        "age": age,
        "schedule": schedule,
        "poster_url": poster,
        "start_date": start_date if start_date else None,
        "end_date": end_date if end_date else None,
        "combined_text": combined_text,
    }


def create_index(es: Elasticsearch):
    """ES 인덱스 생성 (기존 인덱스가 있으면 삭제 후 재생성)"""
    if es.indices.exists(index=INDEX_NAME):
        print(f"⚠️  기존 인덱스 '{INDEX_NAME}' 삭제 중...")
        es.indices.delete(index=INDEX_NAME)

    print(f"📦 인덱스 '{INDEX_NAME}' 생성 중...")
    es.indices.create(index=INDEX_NAME, body=INDEX_MAPPING)
    print(f"✅ 인덱스 '{INDEX_NAME}' 생성 완료!")


def collect_performances() -> list[dict]:
    """KOPIS에서 공연 데이터를 수집하여 ES 문서 리스트로 반환"""
    collected = {}  # performance_id → perf raw data (중복 방지)

    today = datetime.now().strftime("%Y%m%d")
    future = (datetime.now() + timedelta(days=90)).strftime("%Y%m%d")

    print("\n🔍 KOPIS 공연 데이터 수집 시작...")
    print(f"   수집 기간: {today} ~ {future}")

    # 장르별로 공연 목록 수집
    for genre_name, genre_code in GENRE_CODE_MAP.items():
        print(f"\n📋 장르: {genre_name} ({genre_code})")
        try:
            results = fetch_performance_list(
                genre_code=genre_code,
                start_date=today,
                end_date=future,
                rows=50,
            )
            new_count = 0
            for perf in results:
                pid = perf.get("mt20id", "")
                if pid and pid not in collected:
                    perf["shcate"] = genre_code
                    collected[pid] = perf
                    new_count += 1
            print(f"   → {len(results)}건 조회, {new_count}건 신규 수집")
            time.sleep(0.3)
        except Exception as e:
            print(f"   ❌ 오류: {e}")

    print(f"\n📊 총 {len(collected)}건의 고유 공연 수집 완료")

    # 상세 정보 조회 및 ES 문서 생성
    print("\n📝 상세 정보 조회 및 문서 생성 중...")
    documents = []
    for i, (pid, perf) in enumerate(collected.items()):
        try:
            detail = fetch_performance_detail(pid)
            doc = build_document(perf, detail)
            documents.append(doc)
            if (i + 1) % 20 == 0:
                print(f"   → {i + 1}/{len(collected)}건 처리 완료")
            time.sleep(0.2)
        except Exception as e:
            print(f"   ❌ [{pid}] 상세 조회 실패: {e}")
            doc = build_document(perf, {})
            documents.append(doc)

    print(f"\n✅ 총 {len(documents)}건의 문서 생성 완료")
    return documents


def index_documents(es: Elasticsearch, openai_client: OpenAI, documents: list[dict]):
    """문서에 임베딩을 추가하고 ES에 Bulk 인덱싱"""
    # 임베딩 생성
    print(f"\n🧠 OpenAI 임베딩 생성 중 ({EMBEDDING_MODEL})...")
    texts = [doc["combined_text"] for doc in documents]
    embeddings = generate_embeddings(openai_client, texts)
    print(f"✅ {len(embeddings)}건의 임베딩 생성 완료 (차원: {EMBEDDING_DIMS})")

    # 문서에 임베딩 추가
    for doc, emb in zip(documents, embeddings):
        doc["embedding"] = emb

    # Bulk 인덱싱
    print(f"\n⬆️  ES 인덱싱 시작 (Bulk)...")
    actions = [
        {
            "_index": INDEX_NAME,
            "_id": doc["performance_id"],
            "_source": doc,
        }
        for doc in documents
    ]

    success, errors = bulk(es, actions, raise_on_error=False, refresh=True)
    print(f"✅ 인덱싱 완료: 성공 {success}건")
    if errors:
        print(f"⚠️  실패 {len(errors)}건")
        for err in errors[:5]:
            print(f"   → {err}")

    return success


def main():
    print("=" * 60)
    print("🎭 KOPIS → Elasticsearch 공연 데이터 인덱싱")
    print(f"   임베딩 모델: {EMBEDDING_MODEL} ({EMBEDDING_DIMS}차원)")
    print("=" * 60)

    es = get_es_client()
    openai_client = get_openai_client()

    # ES 연결 확인
    info = es.info()
    print(f"✅ ES 연결: {info['cluster_name']} (v{info['version']['number']})")

    # 인덱스 생성
    create_index(es)

    # 데이터 수집
    documents = collect_performances()

    # 임베딩 생성 및 인덱싱
    index_documents(es, openai_client, documents)

    # 결과 확인
    count = es.count(index=INDEX_NAME)["count"]
    print(f"\n{'=' * 60}")
    print(f"🎉 인덱싱 완료! '{INDEX_NAME}' 인덱스에 {count}건 저장됨")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
