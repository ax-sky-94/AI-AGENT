from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from app.agents.prompts import system_prompt
from app.agents.tools import (
    search_performances,
    get_performance_detail,
    recommend_performances,
    ChatResponse,
)
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
