# 공연/전시회 AI 에이전트 설계

## 개요
KOPIS(공연예술통합전산센터) API를 활용한 공연/전시회 정보 제공 AI 에이전트

## 에이전트 페르소나
- 역할: 공연/전시회 정보 전문 AI 어시스턴트 ("공연 도우미")
- 기능: 공연 검색, 추천, 상세정보 제공
- 제한: 예매 대행 불가, 정보 제공만

## Tools (3개)

### 1. search_performances
- 공연 목록 검색 (기간, 장르, 지역 필터)
- KOPIS `get_performance_list()` 활용

### 2. get_performance_detail
- 특정 공연 상세정보 조회 (가격, 시간, 출연진 등)
- KOPIS 공연 상세 API 활용

### 3. recommend_performances
- 장르/지역 기반 추천 (현재 공연 중인 것 위주)
- KOPIS `get_performance_list()` + 필터링

## 수정/생성 파일

| 파일 | 작업 | 설명 |
|------|------|------|
| `agents/prompts.py` | 수정 | 공연 도우미 시스템 프롬프트 |
| `agents/tools.py` | 신규 | KOPIS API 기반 3개 Tool |
| `agents/performance_agent.py` | 신규 | LangGraph 에이전트 생성 |
| `services/agent_service.py` | 수정 | performance_agent 연결 |
| `pyproject.toml` | 수정 | kopisapi 의존성 추가 |

## 데이터 소스
- KOPIS API (`kopisapi` 라이브러리)
- 내일: ElasticSearch 연동 예정

## 데이터 흐름
```
사용자 질문 → LLM 의도 파악 → Tool 선택 → KOPIS API 호출 → LLM 응답 정리 → SSE 스트리밍
```
