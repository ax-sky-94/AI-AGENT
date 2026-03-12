# 공연/전시회 AI 에이전트 구현 계획

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** KOPIS API를 활용하여 공연/전시회 검색, 추천, 상세정보를 제공하는 LangGraph 기반 AI 에이전트 구현

**Architecture:** LangGraph의 `create_react_agent`로 ReAct 에이전트를 생성하고, 3개의 KOPIS Tool을 바인딩한다. 에이전트는 사용자 질문을 파악하여 적절한 Tool을 호출하고, 결과를 자연어로 정리하여 `ChatResponse` Tool로 최종 응답한다.

**Tech Stack:** LangChain v1.0, LangGraph, langchain-openai, kopisapi, FastAPI

---

### Task 1: 의존성 및 환경변수 추가

**Files:**
- Modify: `pyproject.toml:6-16`
- Modify: `env.sample`
- Modify: `.env`
- Modify: `app/core/config.py:15-41`

**Step 1: pyproject.toml에 의존성 추가**

`kopisapi`와 `langgraph` 패키지를 dependencies에 추가한다.

```toml
dependencies = [
    "fastapi>=0.104.0",
    "uvicorn[standard]>=0.24.0",
    "langchain>=1.0.0",
    "langchain-openai>=1.0.0",
    "langgraph>=0.2.0",
    "pydantic>=2.0.0",
    "pydantic-settings>=2.0.0",
    "python-dotenv>=1.0.0",
    "httpx>=0.25.0",
    "opik>=1.10.4",
    "kopisapi>=0.1.0",
]
```

**Step 2: env.sample에 KOPIS_API_KEY 추가**

```
# KOPIS API 설정
KOPIS_API_KEY=your_kopis_api_key
```

**Step 3: .env에도 동일하게 KOPIS_API_KEY 추가**

실제 발급받은 KOPIS API 키를 넣는다. (https://www.kopis.or.kr/por/cs/openapi/openApiInfo.do 에서 발급)

**Step 4: config.py에 KOPIS_API_KEY 설정 추가**

```python
class Settings(BaseSettings):
    # ... 기존 필드 유지 ...

    # KOPIS API 설정
    KOPIS_API_KEY: str = ""
```

**Step 5: 의존성 설치**

Run: `cd /Users/song-kyung-yoon/ai_agent/agent && uv sync`
Expected: 의존성 설치 완료

---

### Task 2: Tools 구현 (tools.py)

**Files:**
- Create: `app/agents/tools.py`

3개의 LangChain Tool을 구현한다. 각 Tool은 `@tool` 데코레이터를 사용하며, KOPIS API를 통해 데이터를 조회한다.

**Step 1: tools.py 작성**

```python
from langchain_core.tools import tool
from kopisapi import KopisAPI
from app.core.config import settings
from datetime import datetime, timedelta


def _get_kopis() -> KopisAPI:
    """KOPIS API 클라이언트 생성"""
    return KopisAPI(settings.KOPIS_API_KEY)


# 지역 매핑 (한글 -> kopisapi 파라미터)
REGION_MAP = {
    "서울": "seoul",
    "부산": "busan",
    "대구": "daegu",
    "인천": "inchon",
    "광주": "gwangju",
    "대전": "daejeon",
    "울산": "ulsan",
    "세종": "sejong",
    "경기": "gyeonggi",
    "강원": "gangwon",
    "충북": "chungbuk",
    "충남": "chungnam",
    "전북": "jeonbuk",
    "전남": "jeonnam",
    "경북": "gyeongbuk",
    "경남": "gyeongnam",
    "제주": "jeju",
}

# 장르 매핑 (한글 -> kopisapi 파라미터)
GENRE_MAP = {
    "연극": "act",
    "뮤지컬": "musical",
    "무용": "dance",
    "클래식": "classic",
    "오페라": "opera",
    "국악": "korean_classical",
    "복합": "complex",
}


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
        kopis = _get_kopis()

        if not start_date:
            start_date = datetime.now().strftime("%Y%m%d")
        if not end_date:
            end_date = (datetime.now() + timedelta(days=30)).strftime("%Y%m%d")

        kwargs = {
            "start_date": start_date,
            "end_date": end_date,
        }

        if genre and genre in GENRE_MAP:
            kwargs["genre"] = GENRE_MAP[genre]
        if region and region in REGION_MAP:
            kwargs["region"] = REGION_MAP[region]

        results = kopis.get_performance_list(**kwargs)

        if not results:
            return "검색 결과가 없습니다. 다른 조건으로 다시 검색해보세요."

        # DataFrame 결과를 문자열로 변환
        output = f"총 {len(results)}건의 공연이 검색되었습니다.\n\n"
        for idx, row in results.head(10).iterrows():
            output += f"- {row.to_dict()}\n"

        if len(results) > 10:
            output += f"\n... 외 {len(results) - 10}건 더 있습니다."

        return output
    except Exception as e:
        return f"공연 검색 중 오류가 발생했습니다: {str(e)}"


@tool
def get_performance_detail(performance_id: str) -> str:
    """특정 공연의 상세 정보를 조회합니다. 공연 ID가 필요합니다.

    Args:
        performance_id: 공연 ID (예: PF12345)

    Returns:
        공연 상세 정보 (제목, 기간, 장소, 가격, 출연진 등)
    """
    try:
        kopis = _get_kopis()
        detail = kopis.get_performance_detail(performance_id)

        if detail is None or (hasattr(detail, 'empty') and detail.empty):
            return f"공연 ID '{performance_id}'에 대한 정보를 찾을 수 없습니다."

        return f"공연 상세 정보:\n{detail.to_dict() if hasattr(detail, 'to_dict') else str(detail)}"
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
        kopis = _get_kopis()

        today = datetime.now().strftime("%Y%m%d")
        week_later = (datetime.now() + timedelta(days=7)).strftime("%Y%m%d")

        kwargs = {
            "start_date": today,
            "end_date": week_later,
        }

        if genre and genre in GENRE_MAP:
            kwargs["genre"] = GENRE_MAP[genre]
        if region and region in REGION_MAP:
            kwargs["region"] = REGION_MAP[region]

        results = kopis.get_performance_list(**kwargs)

        if not results or (hasattr(results, 'empty') and results.empty):
            return "현재 추천할 공연이 없습니다. 다른 장르나 지역으로 시도해보세요."

        output = f"🎭 {region} 지역 추천 공연 (이번 주)\n\n"
        for idx, row in results.head(5).iterrows():
            output += f"- {row.to_dict()}\n"

        return output
    except Exception as e:
        return f"공연 추천 중 오류가 발생했습니다: {str(e)}"
```

**Step 2: 임포트 확인**

Run: `cd /Users/song-kyung-yoon/ai_agent/agent && uv run python -c "from app.agents.tools import search_performances, get_performance_detail, recommend_performances; print('Tools imported OK')"`
Expected: `Tools imported OK`

---

### Task 3: 시스템 프롬프트 수정 (prompts.py)

**Files:**
- Modify: `app/agents/prompts.py`

**Step 1: prompts.py 전체 교체**

```python
system_prompt = """당신은 공연/전시회 정보 전문 AI 어시스턴트 "공연 도우미"입니다.
사용자에게 공연과 전시회 정보를 검색하고, 추천하고, 상세 정보를 제공합니다.

# 역할:
- 공연/전시회 검색: 사용자가 원하는 조건(장르, 지역, 기간, 키워드)에 맞는 공연을 검색합니다.
- 공연 추천: 사용자의 취향에 맞는 현재 진행 중인 공연을 추천합니다.
- 상세 정보 제공: 특정 공연의 가격, 일정, 장소, 출연진 등 상세 정보를 제공합니다.

# 사용 가능한 도구:
1. search_performances: 공연 목록 검색 (키워드, 장르, 지역, 기간 필터링)
2. get_performance_detail: 특정 공연 상세 정보 조회 (공연 ID 필요)
3. recommend_performances: 장르/지역 기반 공연 추천

# 지원 장르: 연극, 뮤지컬, 무용, 클래식, 오페라, 국악, 복합
# 지원 지역: 서울, 부산, 대구, 인천, 광주, 대전, 울산, 세종, 경기, 강원, 충북, 충남, 전북, 전남, 경북, 경남, 제주

# 응답 지침:
- 검색 결과를 보기 좋게 정리하여 제공합니다.
- 사용자가 날짜를 자연어로 말하면 (예: "이번 주말", "다음 달") 적절한 날짜로 변환합니다.
- 예매 대행은 불가하며, 정보 제공만 가능합니다.
- 모르는 정보에 대해서는 솔직하게 모른다고 답변합니다.

# Response Format:
반드시 ChatResponse 도구를 사용하여 최종 응답을 반환하세요.
{
    "message_id (필수 키값)": "<생성된 message_id (UUID 형식)>",
    "content (필수 키값)": "<사용자 질문에 대한 답변>",
    "metadata (필수 키값)": {}
}
"""
```

---

### Task 4: 에이전트 생성 (performance_agent.py)

**Files:**
- Create: `app/agents/performance_agent.py`

LangGraph의 `create_react_agent`를 사용하여 에이전트를 생성한다.

**Step 1: performance_agent.py 작성**

```python
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from app.agents.prompts import system_prompt
from app.agents.tools import (
    search_performances,
    get_performance_detail,
    recommend_performances,
)
from app.models.chat import ChatResponse
from app.core.config import settings


def create_performance_agent(checkpointer=None):
    """공연/전시회 에이전트를 생성합니다."""
    llm = ChatOpenAI(
        model=settings.OPENAI_MODEL,
        api_key=settings.OPENAI_API_KEY,
        temperature=0,
    )

    tools = [
        search_performances,
        get_performance_detail,
        recommend_performances,
        ChatResponse,
    ]

    if checkpointer is None:
        checkpointer = MemorySaver()

    agent = create_react_agent(
        llm,
        tools,
        checkpointer=checkpointer,
        prompt=system_prompt,
    )

    return agent
```

Note: `ChatResponse`는 `app/models/chat.py`에 Pydantic 모델로 정의되어 있으나, LangGraph에서 Tool로 사용하려면 별도 처리가 필요할 수 있다. agent_service.py에서 tool_call name이 "ChatResponse"인 것을 감지하여 최종 응답으로 처리하는 패턴이므로, `ChatResponse`를 LangChain Tool로 변환해야 한다.

**Step 2: ChatResponse를 Tool로 변환**

`app/agents/tools.py` 하단에 추가:

```python
@tool
def ChatResponse(message_id: str, content: str, metadata: dict = {}) -> str:
    """최종 응답을 사용자에게 전달하기 위한 도구입니다. 반드시 이 도구를 사용하여 최종 답변을 반환하세요.

    Args:
        message_id: 고유 메시지 ID (UUID 형식)
        content: 사용자에게 전달할 최종 답변 내용
        metadata: 추가 메타데이터 (기본값: 빈 딕셔너리)

    Returns:
        최종 응답 확인 메시지
    """
    return f"Response delivered: {content[:100]}..."
```

그리고 `performance_agent.py`에서는 이 Tool을 import:
```python
from app.agents.tools import (
    search_performances,
    get_performance_detail,
    recommend_performances,
    ChatResponse,
)
```

(chat.py의 ChatResponse Pydantic 모델 import 제거)

---

### Task 5: agent_service.py 수정

**Files:**
- Modify: `app/services/agent_service.py:14-26`

**Step 1: AgentService 클래스의 __init__과 _create_agent 수정**

`_create_agent`에서 dummy Agent 대신 `create_performance_agent`를 호출한다.

```python
class AgentService:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.OPENAI_MODEL,
            api_key=settings.OPENAI_API_KEY,
            temperature=0,
        )
        self.checkpointer = MemorySaver()
        self.agent = None
        self.progress_queue: asyncio.Queue = asyncio.Queue()

    def _create_agent(self, thread_id: uuid.UUID = None):
        """LangGraph 공연/전시회 에이전트 생성"""
        from app.agents.performance_agent import create_performance_agent
        self.agent = create_performance_agent(checkpointer=self.checkpointer)
```

필요한 import 추가:
```python
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from app.core.config import settings
```

---

### Task 6: 통합 테스트

**Step 1: 서버 실행 확인**

Run: `cd /Users/song-kyung-yoon/ai_agent/agent && uv run uvicorn app.main:app --host 0.0.0.0 --port 8000`
Expected: 서버 정상 기동

**Step 2: API 호출 테스트**

Run:
```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"thread_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6", "message": "서울에서 이번 주에 볼 수 있는 뮤지컬 추천해줘"}'
```
Expected: SSE 스트리밍으로 공연 추천 결과 반환

**Step 3: 다양한 시나리오 테스트**

- "오페라의 유령 공연 정보 알려줘" → search_performances 호출
- "이번 달 부산 클래식 공연 있어?" → search_performances (지역+장르 필터)
- "이 공연 상세 정보 알려줘" → get_performance_detail 호출

---

## 구현 순서 요약

```
Task 1: 의존성/환경변수 → Task 2: tools.py → Task 3: prompts.py → Task 4: performance_agent.py → Task 5: agent_service.py → Task 6: 테스트
```

## 참고 사항

- KOPIS API 키는 https://www.kopis.or.kr/por/cs/openapi/openApiInfo.do 에서 발급
- `kopisapi` 라이브러리가 반환하는 데이터가 DataFrame인지 dict인지 실행 시 확인 필요 → 필요에 따라 tools.py의 결과 파싱 로직 조정
- 내일: ElasticSearch 연동 시 tools.py의 API 직접 호출 부분만 ES Retriever로 교체하면 됨
