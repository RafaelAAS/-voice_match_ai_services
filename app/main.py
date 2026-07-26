from fastapi import FastAPI

from app.api.routes import router as ai_router

app = FastAPI(
    title="VoiceMatch AI Services",
    description="Microservice for handling AI operations (STT, LLM, TTS)",
    version="1.0.0",
)

app.include_router(ai_router)


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
