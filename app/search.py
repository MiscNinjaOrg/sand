import asyncio
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
import nest_asyncio
from pydantic import BaseModel
nest_asyncio.apply()

load_dotenv()

search_app = FastAPI()

class SerpResponse(BaseModel):
    title: str
    link: str
    snippet: str

class Query(BaseModel):
    query: str
    sources: list[SerpResponse]

async def format_sources(sources: list[SerpResponse]) -> str:
    sources_formatted = ""
        
    for i, source in enumerate(sources):
        sources_formatted += "Source " + str(i+1) + ":\n"
        sources_formatted += "Title: " + source.title + "\n"
        sources_formatted += source.snippet + "\n\n"
    return sources_formatted

search_template = """
I want to know about the following query. Use the following sources to answer the given query as best as possible. Be original, concise, accurate and helpful. Wherever applicable, cite the articles as sources only after the relevant sentences in square brackets (e.g. sentence [1][2]).

Query: {query}

{sources}
"""

search_prompt = PromptTemplate(
    template=search_template,
    input_variables=['query', 'sources'],
)

async def streamer(model: str, query: str, sources: list[SerpResponse]):

    callback_handler = AsyncIteratorCallbackHandler()

    gpt_3_5_turbo_llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, streaming=True, callbacks=[callback_handler])
    gpt_4_llm = ChatOpenAI(model_name="gpt-4", temperature=0, streaming=True, callbacks=[callback_handler])

    MODEL_TO_LLM = {
        "gpt-3.5-turbo": gpt_3_5_turbo_llm,
        "gpt-4": gpt_4_llm
    }

    search_chain = LLMChain(
        llm=MODEL_TO_LLM[model],
        prompt=search_prompt,
    )
    run = asyncio.create_task(search_chain.arun(query=query, sources=sources))
    async for token in callback_handler.aiter():
        yield token
    await run

@search_app.post("/{model}")
async def search(
    model: str,
    q: Query
):
    query = q.query
    sources = q.sources
    sources = await format_sources(sources)
    print(sources)

    return StreamingResponse(streamer(model, query, sources), media_type='text/event-stream')