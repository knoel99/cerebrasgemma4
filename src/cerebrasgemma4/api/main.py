from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from cerebrasgemma4.api.routes import convert, jobs

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(
    title="Vid2Doc",
    description="Convert video to Markdown using Gemma 4 31B on Cerebras",
    version="0.1.0",
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.include_router(convert.router)
app.include_router(jobs.router)


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "model": "gemma-4-31b"}