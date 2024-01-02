import asyncio
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
import nest_asyncio
from pydantic import BaseModel
nest_asyncio.apply()

load_dotenv()

chat_app = FastAPI()

class Message(BaseModel):
    name: str
    text: str

class Query(BaseModel):
    messages: list[Message]
    prompt: str

async def format_messages(messages: Message, prompt: str):
    out = []
    for message in messages:
        if message.name == "human":
            out.append(HumanMessage(content=message.text))
        elif message.name == "ai":
            out.append(AIMessage(content=message.text))
    out.append(HumanMessage(content=prompt))
    return out

async def streamer(model: str, messages: list):

    callback_handler = AsyncIteratorCallbackHandler()

    gpt_3_5_turbo_llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, streaming=True, callbacks=[callback_handler])
    gpt_4_llm = ChatOpenAI(model_name="gpt-4", temperature=0, streaming=True, callbacks=[callback_handler])

    MODEL_TO_LLM = {
        "gpt-3.5-turbo": gpt_3_5_turbo_llm,
        "gpt-4": gpt_4_llm
    }

    llm = MODEL_TO_LLM[model] 
    run = asyncio.create_task(llm.ainvoke(messages))
    async for token in callback_handler.aiter():
        yield token
    await run

@chat_app.post("/{model}")
async def search(
    model: str,
    q: Query
):
    messages = q.messages
    prompt = q.prompt
    messages = await format_messages(messages, prompt)
    print(messages)

    return StreamingResponse(streamer(model, messages), media_type='text/event-stream')