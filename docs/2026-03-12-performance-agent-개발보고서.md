# 공연/전시회 AI Agent 개발 보고서

> **작성일**: 2026-03-12
> **프로젝트**: ai_agent (Agent Education Template Server)
> **도메인**: 공연/전시회 정보 검색, 추천, 상세 조회
> **데이터 소스**: KOPIS (공연예술통합전산센터) 공공 API

---

## 1. 개요

사내 AI Agent 교육 과정에서 LangChain + LangGraph 기반의 ReAct Agent를 구현했습니다.
기존 템플릿 프로젝트(`agent-edu-template-server`)를 기반으로 **공연/전시회 도우미** 에이전트를 개발하고, 프론트엔드 UI(`ai_agent_ui`)와 연동하여 실제 서비스 형태로 동작하는 것까지 완료했습니다.

### 오늘의 목표
- Agent 개발 (도구 구현 + LangGraph 에이전트 생성 + 서비스 연동)
- UI 연동 및 테스트 완료
- 내일: ElasticSearch 연동 (Retriever로 전환)

---

## 2. 기술 스택

| 구분 | 기술 | 버전 |
|------|------|------|
| Backend Framework | FastAPI + Uvicorn | 0.104+ |
| AI Framework | LangChain + LangGraph | 1.0+ / 0.2+ |
| LLM | OpenAI GPT (ChatOpenAI) | - |
| 데이터 소스 | KOPIS 공공 API | REST XML |
| XML 파싱 | xmltodict | - |
| Frontend | React + Vite + TypeScript | - |
| 상태 관리 | Jotai | - |
| 스트리밍 | SSE (Server-Sent Events) | @microsoft/fetch-event-source |

---

## 3. 아키텍처

```
┌─────────────────────────────────────────────────────────┐
│  Frontend (React + Vite, :5173)                         │
│  ┌──────────┐  SSE Stream   ┌────────────────────────┐  │
│  │ Chat UI  │ ◄──────────── │ @microsoft/fetch-event │  │
│  │ (Jotai)  │               │ -source                │  │
│  └──────────┘               └────────────────────────┘  │
└────────────────────────┬────────────────────────────────┘
                         │ POST /api/v1/chat/
┌────────────────────────▼────────────────────────────────┐
│  Backend (FastAPI + Uvicorn, :8000)                      │
│  ┌──────────────────────────────────────────────────┐   │
│  │ AgentService                                      │   │
│  │  ├─ _global_checkpointer (MemorySaver 싱글톤)     │   │
│  │  ├─ process_query() → SSE 스트리밍               │   │
│  │  └─ ChatResponse 도구 인터셉트 → step: "done"    │   │
│  └──────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────┐   │
│  │ LangGraph ReAct Agent (create_react_agent)        │   │
│  │  ├─ LLM: ChatOpenAI (temperature=0)              │   │
│  │  ├─ Checkpointer: MemorySaver (멀티턴 메모리)     │   │
│  │  └─ Tools:                                        │   │
│  │     ├─ search_performances (검색)                 │   │
│  │     ├─ get_performance_detail (상세 조회)          │   │
│  │     ├─ recommend_performances (추천)               │   │
│  │     └─ ChatResponse (최종 응답 전달)               │   │
│  └───────────────────────┬──────────────────────────┘   │
└──────────────────────────┬──────────────────────────────┘
                           │ HTTP GET (XML)
┌──────────────────────────▼──────────────────────────────┐
│  KOPIS 공공 API (http://www.kopis.or.kr/openApi/)       │
│  ├─ /pblprfr (공연 목록 검색)                            │
│  └─ /pblprfr/{id} (공연 상세 조회)                       │
└─────────────────────────────────────────────────────────┘
```

---

## 4. 구현 내용

### 4.1 신규 생성 파일

#### (1) `app/agents/tools.py` - KOPIS API 도구 4개

| 도구 | 기능 | 주요 파라미터 |
|------|------|-------------|
| `search_performances` | 공연 검색 | keyword, genre, region, start_date, end_date |
| `get_performance_detail` | 상세 조회 | performance_id |
| `recommend_performances` | 공연 추천 | genre, region |
| `ChatResponse` | 최종 응답 전달 | message_id, content, metadata |

**핵심 구현 사항:**
- KOPIS API 직접 HTTP 호출 (`requests` + `xmltodict`)
- 지역/장르 한글 → API 코드 매핑 딕셔너리
- `_kopis_request()`: 공통 API 호출 함수 (인증키 자동 삽입, XML 파싱)
- `_format_performance()`: 공연 데이터 포맷팅

```python
# 지역 코드 매핑 예시
REGION_CODE_MAP = {
    "서울": "11", "부산": "26", "대구": "27", ...
}

# 장르 코드 매핑 예시
GENRE_CODE_MAP = {
    "연극": "AAAA", "뮤지컬": "GGGA", "클래식": "CCCA", ...
}
```

#### (2) `app/agents/performance_agent.py` - LangGraph ReAct 에이전트

```python
def create_performance_agent(checkpointer=None):
    llm = ChatOpenAI(model=settings.OPENAI_MODEL, temperature=0)
    tools = [search_performances, get_performance_detail,
             recommend_performances, ChatResponse]
    agent = create_react_agent(llm, tools, checkpointer=checkpointer,
                                prompt=system_prompt)
    return agent
```

### 4.2 수정 파일

#### (1) `app/agents/prompts.py` - 시스템 프롬프트

- 페르소나: "공연 도우미" AI 어시스턴트
- 역할 정의: 검색, 추천, 상세 정보 제공
- 지원 장르/지역 명시
- ChatResponse 도구를 통한 최종 응답 포맷 지정

#### (2) `app/services/agent_service.py` - 서비스 레이어 (핵심 수정)

**주요 변경 사항 3가지:**

| 변경 | 이유 | 해결 방법 |
|------|------|----------|
| `_global_checkpointer` 싱글톤 | 매 요청마다 AgentService가 새로 생성 → 메모리 초기화 | 모듈 레벨 전역 변수로 MemorySaver 유지 |
| LangGraph v2 step 이름 호환 | v2에서 LLM 노드 이름이 `"model"` → `"agent"`로 변경 | `step in ["model", "agent", "tools"]` |
| ChatResponse tools step 스킵 | `done` 이벤트 이후 tools step이 isLoading을 다시 true로 | `if message.name == "ChatResponse": continue` |

#### (3) `app/core/config.py` - 설정

- `KOPIS_API_KEY: str = ""` 추가

#### (4) `pyproject.toml` - 의존성

- `langgraph>=0.2.0` 추가
- `kopisapi>=0.1.0` 추가 (직접 사용하지 않으나 참조용)

#### (5) `.env` - 환경변수

- `KOPIS_API_KEY` 추가

---

## 5. 문제 해결 기록

개발 과정에서 발생한 주요 이슈와 해결 과정입니다.

### 5.1 kopisapi 라이브러리 버그

| 항목 | 내용 |
|------|------|
| **증상** | kopisapi 라이브러리 사용 시 KOPIS API 오류 |
| **원인** | 라이브러리가 `rows=500`으로 요청하나, API 최대 조회 건수는 100건 |
| **에러** | `"최대 조회수는 100건까지 가능합니다"` |
| **해결** | kopisapi 라이브러리 대신 `requests` + `xmltodict`로 직접 HTTP 호출, `rows=20` 설정 |

### 5.2 LangGraph v2 노드 이름 변경

| 항목 | 내용 |
|------|------|
| **증상** | 에이전트 응답이 UI에 표시되지 않음 (SSE `done` 이벤트 미발생) |
| **원인** | LangGraph v2에서 LLM 노드 이름이 `"model"` → `"agent"`로 변경 |
| **해결** | `agent_service.py`의 step 필터에 `"agent"` 추가 |

```python
# Before
if not event or not (step in ["model", "tools"]):
# After
if not event or not (step in ["model", "agent", "tools"]):
```

### 5.3 UI 크래시 (CodeMirror 에러)

| 항목 | 내용 |
|------|------|
| **증상** | `"value must be typeof string but got object"` |
| **원인** | tools step의 `message.content`가 JSON 문자열이 아닌 dict 객체로 전송 |
| **해결** | `json.dumps(message.content, ensure_ascii=False)` 로 직렬화 |

### 5.4 UI 로딩 무한 대기

| 항목 | 내용 |
|------|------|
| **증상** | UI가 "ChatResponse 처리 중입니다."에서 멈춤 |
| **원인** | `step: "done"` 이후 `step: "tools", name: "ChatResponse"` 이벤트가 추가 전송 → isLoading이 다시 true |
| **해결** | tools step에서 ChatResponse는 건너뜀 (`if message.name == "ChatResponse": continue`) |

### 5.5 멀티턴 메모리 미작동

| 항목 | 내용 |
|------|------|
| **증상** | 후속 질문에서 이전 대화 맥락을 기억하지 못함 |
| **원인** | `chat.py`에서 매 요청마다 `AgentService()` 새로 생성 → `MemorySaver()` 초기화 |
| **해결** | `MemorySaver`를 모듈 레벨 전역 싱글톤 (`_global_checkpointer`)으로 변경 |

---

## 6. SSE 스트리밍 흐름

프론트엔드-백엔드 간 SSE 이벤트 흐름입니다.

```
사용자 입력: "서울 뮤지컬 추천해줘"

[Backend → Frontend SSE Events]

1. {"step": "model", "tool_calls": ["recommend_performances"]}
   → UI: "recommend_performances 처리 중입니다."

2. {"step": "tools", "name": "recommend_performances", "content": "...검색결과..."}
   → UI: "recommend_performances 처리 중입니다." (계속)

3. {"step": "done", "message_id": "...", "content": "서울에서 추천하는 뮤지컬...", ...}
   → UI: 최종 응답 표시, isLoading = false

※ ChatResponse tools step은 서버에서 스킵 (UI 재로딩 방지)
```

---

## 7. 테스트 결과

### 7.1 테스트 방법
- **UI 테스트**: `http://localhost:5173/chat` 에서 브라우저 직접 테스트
- **서버**: FastAPI `:8000` + Vite `:5173`

### 7.2 테스트 결과 요약 (16/16 PASS)

#### 기본 기능 테스트

| # | 테스트 케이스 | 입력 | 결과 |
|---|-------------|------|------|
| 1-1 | 키워드 검색 | "김종욱 찾기 검색해줘" | **PASS** - 상세 정보 반환 |
| 1-2 | 장르+지역 복합 검색 | "서울에서 하는 연극 공연 찾아줘" | **PASS** - 연극 10건 목록 |
| 1-5 | 자연어 기간 지정 | "4월에 하는 뮤지컬 알려줘" | **PASS** - 날짜 변환 후 검색 |
| 1-6 | 공연 추천 | "대구 클래식 공연 추천해줘" | **PASS** - 5건 상세 추천 |
| 1-8 | ID 직접 조회 | "PF254498 상세 정보 알려줘" | **PASS** - 상세 정보 반환 |

#### 멀티턴 대화 (메모리) 테스트

| # | 테스트 케이스 | 시나리오 | 결과 |
|---|-------------|---------|------|
| 2-1 | 목록→상세 | 1)"추천" → 2)"4번 상세 알려줘" | **PASS** - 4번 공연 기억 |
| 2-2 | 조건 변경 | 1)"대구 클래식" → 2)"부산으로 바꿔서" | **PASS** - 장르 유지, 지역 변경 |
| 2-3 | 3턴 대화 | 1)"추천" → 2)"2번 상세" → 3)"가격 얼마?" | **PASS** - 메모리에서 즉답 |

#### 에지 케이스 테스트

| # | 테스트 케이스 | 입력 | 결과 |
|---|-------------|------|------|
| 3-1 | 검색 결과 없음 | "제주에서 하는 오페라 찾아줘" | **PASS** - 없음 안내 + 대안 제안 |
| 3-2 | 미지원 장르 | "힙합 공연 찾아줘" | **PASS** - 에러 안내 + 대안 제안 |
| 3-3 | 미지원 지역 | "미국에서 공연 찾아줘" | **PASS** - 국내 전용 안내 |
| 3-5 | 빈 메시지 | " " (공백) | **PASS** - 인사 응답 |
| 3-6 | 공연 무관 질문 | "오늘 날씨가 어때?" | **PASS** - 역할 범위 안내 |
| 3-7 | 긴 입력 (200자+) | 복잡한 데이트 공연 요청 | **PASS** - 조건 파싱 후 추천 |

#### UI 연동 테스트

| # | 테스트 케이스 | 결과 |
|---|-------------|------|
| 4-5 | isLoading 해제 | **PASS** - 모든 응답 후 로딩 정상 해제 |
| 5-4 | 동시 요청 (연타) | **PASS** - isTyping 가드로 중복 차단 |

---

## 8. 프로젝트 구조

```
ai_agent/agent/
├── .env                          # KOPIS_API_KEY 등 환경변수
├── pyproject.toml                # 의존성 (langgraph 추가)
├── app/
│   ├── agents/
│   │   ├── tools.py              # [신규] KOPIS API 도구 4개
│   │   ├── prompts.py            # [수정] 공연 도우미 시스템 프롬프트
│   │   └── performance_agent.py  # [신규] LangGraph ReAct 에이전트
│   ├── services/
│   │   └── agent_service.py      # [수정] 스트리밍 + 메모리 연동
│   ├── core/
│   │   └── config.py             # [수정] KOPIS_API_KEY 설정 추가
│   └── ...
└── docs/
    └── 2026-03-12-performance-agent-개발보고서.md  # 본 문서
```

---

## 9. 향후 계획

| 일정 | 작업 | 내용 |
|------|------|------|
| Day 2 | ElasticSearch 연동 | tools.py의 직접 API 호출 → ES Retriever로 전환 |
| Day 2 | 데이터 인덱싱 | KOPIS 공연 데이터를 ES에 주기적 색인 |
| 추후 | 성능 개선 | 응답 속도 최적화, 캐싱 도입 |
| 추후 | 기능 확장 | 예매 링크 연동, 유사 공연 추천, 리뷰 정보 |

---

## 10. 참고사항

- **KOPIS API 제한**: 한 번에 최대 100건 조회 가능 (현재 20건으로 설정)
- **MemorySaver**: 인메모리 방식으로 서버 재시작 시 대화 이력 초기화
- **LangGraph v2 호환**: `"agent"` 노드 이름 사용 (v1의 `"model"`과 함께 지원)
- **SSL 검증 비활성화**: KOPIS API 호출 시 `verify=False` (개발 환경 한정)
