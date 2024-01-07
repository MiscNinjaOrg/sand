import asyncio
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.embeddings import FakeEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
import nest_asyncio
from pydantic import BaseModel
from readabilipy import simple_json_from_html_string
nest_asyncio.apply()

load_dotenv(".env.local")

chat_app = FastAPI()

class Message(BaseModel):
    role: str
    content: str

class Query(BaseModel):
    messages: list[Message]
    vectorStore: str
    pageURL: str

async def format_messages(messages: list[Message], vectorStore: str, pageURL: str):

    out = []

    if vectorStore != "":
        db = FAISS.deserialize_from_bytes(embeddings=OpenAIEmbeddings(), serialized=bytes.fromhex(vectorStore))
        query = messages[-1].content
        docs = db.similarity_search(query)
        docs = docs[:min(5, len(docs))]

        text = " ".join([doc.page_content for doc in docs])

        out.append(SystemMessage(
            content="""
                You are a kind, helpful assistant. The user is working on a page with the following content. Answer all their queries honestly and directly. If necessary and relevant, use the provided page content.

                Page Content: {text}
            """.format(text=text[:min(2000, len(text))])
        ))

    for message in messages:
        if message.role == "human":
            out.append(HumanMessage(content=message.content))
        elif message.role == "ai":
            out.append(AIMessage(content=message.content))
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
async def chat(
    model: str,
    q: Query
):
    messages = q.messages
    vectorStore = q.vectorStore
    pageURL = q.pageURL
    messages = await format_messages(messages, vectorStore, pageURL)
    print(messages)

    return StreamingResponse(streamer(model, messages), media_type='text/event-stream')