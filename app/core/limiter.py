# SEC: Rate Limiting — IP 기반 요청 빈도 제한 (DoS/비용 폭증 방지)
# 앱 전체에서 하나의 Limiter 인스턴스를 공유하여 정확한 카운팅을 보장합니다.
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
