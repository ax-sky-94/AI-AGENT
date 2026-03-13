import uuid

from app.utils.logger import custom_logger
from fastapi import APIRouter, HTTPException, Request
from app.models.chat import ChatRequest
from app.services.agent_service import AgentService
from fastapi.responses import StreamingResponse
from slowapi import Limiter
from slowapi.util import get_remote_address

chat_router = APIRouter()
limiter = Limiter(key_func=get_remote_address)


# SEC: Rate Limiting — /chat 엔드포인트 분당 20회 제한 (OpenAI 비용 폭증/DoS 방지)
@chat_router.post("/chat")
@limiter.limit("20/minute")
async def post_chat(request: Request, chat_request: ChatRequest):
    """
    자연어 쿼리를 에이전트가 처리합니다.

    ## 실제 테스트용 Request json
    ```json
    {
        "thread_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
        "message": "안녕하세요, 오늘 날씨가 어때요?"
    }
    ```
    """
    custom_logger.info(f"API Request: {chat_request}")
    try:
        thread_id = getattr(chat_request, "thread_id", uuid.uuid4())

        async def event_generator():
            try:
                yield f'data: {{"step": "model", "tool_calls": ["Planning"]}}\n\n'
                agent_service = AgentService()
                async for chunk in agent_service.process_query(
                    user_messages=chat_request.message,
                    thread_id=thread_id
                ):
                    yield f"data: {chunk}\n\n"
            except Exception as e:
                # 스트리밍 중 예외 발생 시 에러 메시지를 스트리밍으로 전송
                import json
                from datetime import datetime
                # SEC: 내부 에러 상세를 사용자에게 노출하지 않음 (로그에만 기록)
                error_response = {
                    "step": "done",
                    "message_id": str(uuid.uuid4()),
                    "role": "assistant",
                    "content": "요청 처리 중 오류가 발생했습니다. 다시 시도해주세요.",
                    "metadata": {},
                    "created_at": datetime.utcnow().isoformat(),
                }
                yield f"data: {json.dumps(error_response, ensure_ascii=False)}\n\n"
                custom_logger.error(f"Error in event_generator: {e}")
                import traceback
                custom_logger.error(traceback.format_exc())
        
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream"
        )
        
    except Exception as e:
        # 스트리밍 시작 전 예외만 HTTPException으로 처리
        custom_logger.error(f"Error in /chat (before streaming): {e}")
        import traceback
        custom_logger.error(traceback.format_exc())
        # SEC: 내부 에러 상세를 사용자에게 노출하지 않음
        raise HTTPException(status_code=500, detail="서버 내부 오류가 발생했습니다.")

