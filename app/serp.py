from fastapi import FastAPI
from dotenv import load_dotenv
from langchain.utilities import GoogleSerperAPIWrapper
from pydantic import BaseModel

load_dotenv(".env.local")

class SerpResponse(BaseModel):
    title: str
    link: str
    snippet: str

class ImageResponse(BaseModel):
    imageURL: str
    imageLink: str

class Query(BaseModel):
    query: str

serp_app = FastAPI()

@serp_app.post("/serp")
async def serp(
    query: Query,
) -> list[SerpResponse]:
    serp = GoogleSerperAPIWrapper()
    res = serp.results(query)    
    results = []
    for r in res['organic']:
        results.append({'title': r['title'], 'link': r['link'], 'snippet': r['snippet']})

    return results

@serp_app.post("/news")
async def news(
    query: Query,
) -> list[SerpResponse]:
    serp = GoogleSerperAPIWrapper(type="news")
    res = serp.results(query)    
    results = []
    for r in res['news']:
        results.append({'title': r['title'], 'link': r['link'], 'snippet': r['snippet']})

    return results

@serp_app.post("/images")
async def images(
    query: Query,
) -> list[ImageResponse]:
    serp = GoogleSerperAPIWrapper(type="images")
    res = serp.results(query)    
    results = []
    for r in res['images']:
        results.append({'imageURL': r['imageUrl'], 'imageLink': r['link']})

    return results