from typing import Dict, Optional
from pydantic import BaseModel, Field
from uuid import UUID


class ChatRequest(BaseModel):
    thread_id: UUID
    message: str = Field(..., min_length=1, max_length=2000, description="사용자 메시지 (1~2000자)")


class ResponseMetadata(BaseModel):
    pass


class ChatResponse(BaseModel):
    message_id: str
    content: str
    metadata: ResponseMetadata
