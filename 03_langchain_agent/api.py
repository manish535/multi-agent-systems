import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import datetime
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage

# Import agent from agent.py
from agent import agent

load_dotenv()

# ============================================================
# FastAPI app
# ============================================================

app = FastAPI(
    title="AWS Cost Agent API",
    description="AI agent for AWS cost analysis and alerting",
    version="1.0.0"
)

# ============================================================
# Request / Response models
# ============================================================

class ChatRequest(BaseModel):
    message: str
    thread_id: Optional[str] = "default"

class ChatResponse(BaseModel):
    response: str
    thread_id: str
    input_tokens: int
    output_tokens: int
    timestamp: str

# ============================================================
# Routes
# ===========================================================

@app.get("/")
def root():
    return {
        "service": "AWS Cost Agent API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "timestamp": datetime.datetime.now().isoformat()
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        config = {"configurable": {"thread_id": request.thread_id}}

        result = agent.invoke(
            {"messages": [HumanMessage(content=request.message)]},
            config=config
        )

        final = result["messages"][-1].content
        if isinstance(final, list):
            final = final[0].get("text", str(final))

        last_msg = result["messages"][-1]
        input_tokens = 0
        output_tokens = 0
        if hasattr(last_msg, "usage_metadata") and last_msg.usage_metadata:
            input_tokens = last_msg.usage_metadata.get("input_tokens", 0)
            output_tokens = last_msg.usage_metadata.get("output_tokens", 0)

        return ChatResponse(
            response=final,
            thread_id=request.thread_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            timestamp=datetime.datetime.now().isoformat()
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history/{thread_id}")
def get_history(thread_id: str):
    try:
        config = {"configurable": {"thread_id": thread_id}}
        state = agent.get_state(config)
        messages = state.values.get("messages", [])

        history = []
        for msg in messages:
            history.append({
                "role": msg.__class__.__name__.replace("Message", "").lower(),
                "content": msg.content if isinstance(msg.content, str)
                           else str(msg.content),
                "timestamp": datetime.datetime.now().isoformat()
            })

        return {"thread_id": thread_id, "messages": history}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)