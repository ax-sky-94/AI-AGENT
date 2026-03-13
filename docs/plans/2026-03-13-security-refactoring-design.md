# 보안 리팩토링 설계 문서

> **작성일**: 2026-03-13
> **목적**: LangChain 공식 보안 가이드라인 기준 핵심 위험 항목 수정
> **범위**: 교육 과정 시연용 (기존 기능 100% 유지)
> **참고 문서**: https://docs.langchain.com/oss/python/security-policy

---

## LangChain 3대 보안 원칙 대응

| 원칙 | 수정 항목 | 위험도 |
|------|---------|-------|
| 최소 권한 | CORS 제한 (#3) | 🔴 |
| 오용 대비 | 입력 검증 (#1), 에러 마스킹 (#2), Rate Limiting (#4) | 🔴🟠 |
| 심층 방어 | SSL 경고 (#5) + 기존 프롬프트 가드레일 | 🟡 |

---

## 수정 항목 (위험도 순)

### 1. 입력 검증 — `models/chat.py`
- `message: str` → `Field(min_length=1, max_length=2000)`
- 빈 메시지 및 초대형 입력 차단

### 2. 에러 메시지 마스킹 — `chat.py`, `agent_service.py`
- 사용자 응답에서 `"error": str(e)` 필드 제거
- 서버 로그에만 상세 에러 기록 (이미 구현됨)

### 3. CORS 제한 — `config.py`
- 기본값을 `["*"]` → `["http://localhost:5173", "http://localhost:3000"]`으로 변경
- `.env`의 CORS_ORIGINS 설정이 있으면 그것을 우선 사용

### 4. Rate Limiting — `main.py`, `chat.py`, `pyproject.toml`
- `slowapi` 의존성 추가
- `/chat` 엔드포인트에 분당 20회 제한
- 초과 시 429 Too Many Requests

### 5. SSL 검증 경고 — `elasticsearch.py`, `tools.py`
- `verify_certs=False`에 보안 경고 주석 추가
- 프로덕션 전환 시 변경 필요 사항 명시

---

## 수정 파일 목록

| 파일 | 변경 내용 |
|------|---------|
| `app/models/chat.py` | Field 검증 추가 |
| `app/api/routes/chat.py` | error 필드 제거, rate limit 데코레이터 |
| `app/services/agent_service.py` | error 필드 제거 (3곳) |
| `app/core/config.py` | CORS 기본값 변경 |
| `app/main.py` | slowapi 미들웨어 등록 |
| `app/core/elasticsearch.py` | 보안 경고 주석 |
| `app/agents/tools.py` | 보안 경고 주석 |
| `pyproject.toml` | slowapi 의존성 추가 |
