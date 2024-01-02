import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from search import search_app
from serp import serp_app
from chat import chat_app

load_dotenv()

app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'],
)
app.mount("/search", search_app)
app.mount("/serp", serp_app)
app.mount("/chat", chat_app)

@app.get("/test")
async def test():
    return {
        "response": "hi there"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)