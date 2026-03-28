# main.py — FastAPI application entry point
# Component stubs — full implementation added per component.

from fastapi import FastAPI

app = FastAPI(title="Smart Scheduler")


@app.get("/health")
async def health():
    return {"status": "ok", "service": "smart-scheduler"}
