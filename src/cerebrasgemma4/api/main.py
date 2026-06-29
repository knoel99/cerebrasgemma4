from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI

load_dotenv()
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from cerebrasgemma4.api.routes import chat, convert, jobs

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(
    title="Sightline",
    description="Sightline — video to structured documents with Gemma 4 31B on Cerebras",
    version="0.1.0",
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.include_router(convert.router)
app.include_router(jobs.router)
app.include_router(chat.router)


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "model": "gemma-4-31b"}