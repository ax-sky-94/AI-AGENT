from langchain_core.tools import tool
from app.core.config import settings
from datetime import datetime, timedelta
import requests
import xmltodict


KOPIS_BASE_URL = "http://www.kopis.or.kr/openApi/restful/"

# 지역 코드 매핑 (한글 -> KOPIS API 코드)
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

# 장르 코드 매핑 (한글 -> KOPIS API 코드)
GENRE_CODE_MAP = {
    "연극": "AAAA",
    "뮤지컬": "GGGA",
    "무용": "BBBC",
    "클래식": "CCCA",
    "오페라": "CCCC",
    "국악": "CCCD",
    "복합": "EEEA",
}


def _kopis_request(endpoint: str, params: dict) -> list[dict]:
    """KOPIS API에 요청을 보내고 결과 리스트를 반환"""
    url = f"{KOPIS_BASE_URL}{endpoint}"
    params["service"] = settings.KOPIS_API_KEY

    response = requests.get(url, params=params, verify=False)
    response.raise_for_status()

    data = xmltodict.parse(response.text)
    db_data = data.get("dbs", {}).get("db")

    if not db_data:
        return []

    # 단일 결과인 경우 리스트로 변환
    if isinstance(db_data, dict):
        return [db_data]

    return db_data


def _format_performance(perf: dict) -> str:
    """공연 데이터 dict를 읽기 좋은 문자열로 변환"""
    name = perf.get("prfnm", "알 수 없음")
    venue = perf.get("fcltynm", "알 수 없음")
    start = perf.get("prfpdfrom", "")
    end = perf.get("prfpdto", "")
    genre = perf.get("genrenm", "")
    state = perf.get("prfstate", "")
    perf_id = perf.get("mt20id", "")
    return f"[{perf_id}] {name} | {genre} | {venue} | {start}~{end} | {state}"


@tool
def search_performances(
    keyword: str = "",
    genre: str = "",
    region: str = "",
    start_date: str = "",
    end_date: str = "",
) -> str:
    """공연/전시회를 검색합니다. 키워드, 장르, 지역, 기간으로 필터링할 수 있습니다.

    Args:
        keyword: 공연 이름 키워드 (예: "오페라의 유령", "레미제라블")
        genre: 장르 (연극, 뮤지컬, 무용, 클래식, 오페라, 국악, 복합)
        region: 지역 (서울, 부산, 대구, 인천, 광주, 대전, 울산, 세종, 경기, 강원, 충북, 충남, 전북, 전남, 경북, 경남, 제주)
        start_date: 검색 시작 날짜 (YYYYMMDD 형식, 기본값: 오늘)
        end_date: 검색 종료 날짜 (YYYYMMDD 형식, 기본값: 30일 후)

    Returns:
        검색된 공연 목록 정보
    """
    try:
        if not start_date:
            start_date = datetime.now().strftime("%Y%m%d")
        if not end_date:
            end_date = (datetime.now() + timedelta(days=30)).strftime("%Y%m%d")

        params = {
            "stdate": start_date,
            "eddate": end_date,
            "cpage": "1",
            "rows": "20",
        }

        if genre and genre in GENRE_CODE_MAP:
            params["shcate"] = GENRE_CODE_MAP[genre]
        if region and region in REGION_CODE_MAP:
            params["signgucode"] = REGION_CODE_MAP[region]
        if keyword:
            params["shprfnm"] = keyword

        results = _kopis_request("pblprfr", params)

        if not results:
            return "검색 결과가 없습니다. 다른 조건으로 다시 검색해보세요."

        output_lines = [f"총 {len(results)}건의 공연이 검색되었습니다.\n"]
        for perf in results[:10]:
            output_lines.append(f"- {_format_performance(perf)}")

        if len(results) > 10:
            output_lines.append(f"\n... 외 {len(results) - 10}건 더 있습니다.")

        return "\n".join(output_lines)
    except Exception as e:
        return f"공연 검색 중 오류가 발생했습니다: {str(e)}"


@tool
def get_performance_detail(performance_id: str) -> str:
    """특정 공연의 상세 정보를 조회합니다. search_performances 결과에서 얻은 공연 ID가 필요합니다.

    Args:
        performance_id: 공연 ID (예: PF12345). search_performances 결과의 대괄호 안에 있는 ID입니다.

    Returns:
        공연 상세 정보 (제목, 기간, 장소, 가격, 출연진 등)
    """
    try:
        url = f"{KOPIS_BASE_URL}pblprfr/{performance_id}"
        params = {"service": settings.KOPIS_API_KEY}
        response = requests.get(url, params=params, verify=False)
        response.raise_for_status()

        data = xmltodict.parse(response.text)
        detail = data.get("dbs", {}).get("db", {})

        if not detail:
            return f"공연 ID '{performance_id}'에 대한 정보를 찾을 수 없습니다."

        info_lines = [
            f"공연 상세 정보",
            f"",
            f"공연명: {detail.get('prfnm', '알 수 없음')}",
            f"기간: {detail.get('prfpdfrom', '')} ~ {detail.get('prfpdto', '')}",
            f"장소: {detail.get('fcltynm', '알 수 없음')}",
            f"장르: {detail.get('genrenm', '')}",
            f"출연진: {detail.get('prfcast', '정보 없음')}",
            f"제작진: {detail.get('prfcrew', '정보 없음')}",
            f"런타임: {detail.get('prfruntime', '정보 없음')}",
            f"관람연령: {detail.get('prfage', '정보 없음')}",
            f"가격: {detail.get('pcseguidance', '정보 없음')}",
            f"공연시간: {detail.get('dtguidance', '정보 없음')}",
            f"공연상태: {detail.get('prfstate', '')}",
        ]

        return "\n".join(info_lines)
    except Exception as e:
        return f"공연 상세 조회 중 오류가 발생했습니다: {str(e)}"


@tool
def recommend_performances(
    genre: str = "",
    region: str = "서울",
) -> str:
    """현재 공연 중인 작품 중에서 장르와 지역을 기반으로 공연을 추천합니다.

    Args:
        genre: 선호 장르 (연극, 뮤지컬, 무용, 클래식, 오페라, 국악, 복합). 빈 문자열이면 전체 장르.
        region: 선호 지역 (서울, 부산, 대구 등). 기본값: 서울

    Returns:
        추천 공연 목록
    """
    try:
        today = datetime.now().strftime("%Y%m%d")
        week_later = (datetime.now() + timedelta(days=7)).strftime("%Y%m%d")

        params = {
            "stdate": today,
            "eddate": week_later,
            "cpage": "1",
            "rows": "10",
        }

        if genre and genre in GENRE_CODE_MAP:
            params["shcate"] = GENRE_CODE_MAP[genre]
        if region and region in REGION_CODE_MAP:
            params["signgucode"] = REGION_CODE_MAP[region]

        results = _kopis_request("pblprfr", params)

        if not results:
            return "현재 추천할 공연이 없습니다. 다른 장르나 지역으로 시도해보세요."

        genre_label = genre if genre else "전체"
        output_lines = [f"{region} 지역 {genre_label} 장르 추천 공연 (이번 주)\n"]
        for perf in results[:5]:
            output_lines.append(f"- {_format_performance(perf)}")

        if len(results) > 5:
            output_lines.append(f"\n총 {len(results)}건 중 상위 5건을 추천합니다.")

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
