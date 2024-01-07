from fastapi import FastAPI
from dotenv import load_dotenv
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.embeddings import FakeEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from pydantic import BaseModel
from readabilipy import simple_json_from_html_string


load_dotenv(".env.local")

embeddings_app = FastAPI()

class Query(BaseModel):
    pageContent: str

@embeddings_app.post("/")
async def serp(
    query: Query,
) -> str:
    pageContent = query.pageContent
    article = simple_json_from_html_string(pageContent, use_readability=True)
    articleText = "\n".join([t['text'] for t in article['plain_text']])

    text_splitter = CharacterTextSplitter(separator=" ", chunk_size=400, chunk_overlap=100)
    articleTextSplit = text_splitter.create_documents([articleText])
    documents = text_splitter.split_documents(articleTextSplit)
    
    db = FAISS.from_documents(documents, OpenAIEmbeddings())
    results = db.serialize_to_bytes().hex()

    return results