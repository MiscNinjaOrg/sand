from fastapi import FastAPI
from dotenv import load_dotenv
from langchain.utilities import GoogleSerperAPIWrapper
from pydantic import BaseModel

load_dotenv()

class SerpResponse(BaseModel):
    title: str
    link: str
    snippet: str

class Query(BaseModel):
    query: str

serp_app = FastAPI()

@serp_app.post("/")
async def serp(
    query: Query,
) -> list[SerpResponse]:
    serp = GoogleSerperAPIWrapper()
    res = serp.results(query)    
    results = []
    for r in res['organic']:
        results.append({'title': r['title'], 'link': r['link'], 'snippet': r['snippet']})

    return results