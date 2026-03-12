# IMP: LangGraph StateGraph 기반 공연/전시회 AI 에이전트.
# create_react_agent 대신 명시적인 노드(Node)와 엣지(Edge)로 그래프를 구성합니다.
# 이를 통해 에이전트의 실행 흐름을 세밀하게 제어하고, 추후 검증/가드레일 노드를 추가할 수 있습니다.

from typing import Annotated, TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from app.agents.prompts import get_system_prompt
from app.agents.tools import (
    search_performances,
    get_performance_detail,
    recommend_performances,
    ChatResponse,
)
from app.core.config import settings


# ─── State 정의 ──────────────────────────────────────────────────────────────
# IMP: LangGraph의 상태(State)는 그래프의 모든 노드가 공유하는 데이터입니다.
# add_messages reducer를 사용하여 메시지가 누적되도록 합니다.
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ─── 도구 목록 ────────────────────────────────────────────────────────────────
tools = [
    search_performances,
    get_performance_detail,
    recommend_performances,
    ChatResponse,
]


def create_performance_agent(checkpointer=None):
    """StateGraph 기반 공연/전시회 에이전트를 생성합니다.

    그래프 구조:
        [START] → agent → (tool_calls 있으면) → tools → agent → ...
                        → (tool_calls 없으면) → [END]

    노드:
        - agent: LLM이 사용자 메시지를 분석하고 도구 호출 여부를 결정
        - tools: 선택된 도구를 실행하고 결과를 반환

    엣지:
        - agent → tools: LLM이 도구를 호출할 때
        - agent → END: LLM이 최종 응답을 생성할 때
        - tools → agent: 도구 실행 결과를 LLM에 전달
    """
    # LLM 초기화 (도구 바인딩)
    llm = ChatOpenAI(
        model=settings.OPENAI_MODEL,
        api_key=settings.OPENAI_API_KEY,
        temperature=0,
    )
    # IMP: LLM에 도구를 바인딩하여 Function Calling을 지원합니다.
    llm_with_tools = llm.bind_tools(tools)

    # IMP: 시스템 프롬프트를 동적으로 생성하여 현재 날짜 정보를 포함합니다.
    sys_msg = SystemMessage(content=get_system_prompt())

    # ─── 노드 함수 정의 ──────────────────────────────────────────────────
    # IMP: agent 노드 - LLM이 메시지를 분석하고 도구 호출 또는 최종 응답을 결정합니다.
    def agent_node(state: AgentState) -> dict:
        """LLM 에이전트 노드: 사용자 메시지를 분석하고 다음 행동을 결정합니다."""
        messages = [sys_msg] + state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    # IMP: tools 노드 - LangGraph의 ToolNode를 사용하여 도구를 실행합니다.
    tool_node = ToolNode(tools)

    # ─── 조건부 라우팅 함수 ──────────────────────────────────────────────
    # IMP: agent 노드의 출력을 분석하여 다음 노드를 결정하는 라우터입니다.
    # tool_calls가 있으면 tools 노드로, 없으면 END로 라우팅합니다.
    def should_continue(state: AgentState) -> str:
        """에이전트의 마지막 메시지를 보고 도구 호출 여부를 판단합니다."""
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return END

    # ─── 그래프 구성 ────────────────────────────────────────────────────
    # IMP: StateGraph를 사용하여 명시적으로 노드와 엣지를 정의합니다.
    graph = StateGraph(AgentState)

    # 노드 추가
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)

    # 엣지 추가
    graph.set_entry_point("agent")  # 시작점: agent 노드
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")  # tools → agent 루프

    # 체크포인터 설정
    if checkpointer is None:
        checkpointer = MemorySaver()

    # 그래프 컴파일
    compiled_graph = graph.compile(checkpointer=checkpointer)

    return compiled_graph
