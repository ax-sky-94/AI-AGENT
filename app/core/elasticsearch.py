# IMP: Elasticsearch 클라이언트 싱글톤 모듈.
# 앱 전체에서 하나의 ES 클라이언트를 공유하여 커넥션 풀을 효율적으로 관리합니다.
from elasticsearch import Elasticsearch
from app.core.config import settings


def get_es_client() -> Elasticsearch:
    """Elasticsearch 클라이언트를 생성하여 반환합니다."""
    return Elasticsearch(
        settings.ES_URL,
        basic_auth=(settings.ES_USERNAME, settings.ES_PASSWORD),
        verify_certs=False,
        request_timeout=30,
    )


# 전역 싱글톤 ES 클라이언트
es_client = get_es_client()
