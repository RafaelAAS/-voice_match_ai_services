from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.core.config import settings
from app.schemas.ai_schemas import (
    EvaluateAnswerRequest,
    EvaluateAnswerResponse,
    EvaluateAudioRequest,
    EvaluateAudioResponse,
    GenerateQuestionResponse,
    InterviewContext,
)
from app.services.ai_service_base import AIServiceBase
from app.services.groq_ai_service import GroqAIService
from app.services.mock_ai_service import MockAIService

router = APIRouter(prefix="/ai", tags=["AI Operations"])


def get_ai_service() -> AIServiceBase:
    if settings.mock_ai:
        return MockAIService()
    return GroqAIService()


@router.post("/evaluate-audio", response_model=EvaluateAudioResponse)
async def evaluate_audio_multimodal(
    payload: EvaluateAudioRequest, ai_service: AIServiceBase = Depends(get_ai_service)
):
    """
    Lê o áudio do disco físico (volume do Docker), transcreve, avalia
    comportamentalmente e retorna a próxima pergunta em um único JSON
    validado (ver EvaluateAudioResponse em app/schemas/ai_schemas.py).
    """
    try:
        if not hasattr(ai_service, "process_audio_interview"):
            raise HTTPException(
                status_code=501,
                detail="O serviço de IA ativo não suporta processamento "
                "multimodal de áudio.",
            )

        resultado = await ai_service.process_audio_interview(
            audio_file_path=payload.audio_path,
            context=payload.context.model_dump(),
        )
        return resultado

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Erro interno no motor de IA: {str(e)}"
        )


@router.post("/transcribe", response_model=dict)
async def transcribe_audio(
    file: UploadFile = File(...), ai_service: AIServiceBase = Depends(get_ai_service)
):
    """
    Receives an audio file via HTTP and returns the transcribed text.
    """
    transcription = ai_service.transcribe_audio(file.filename)
    return {"transcription": transcription}


@router.post("/generate-question", response_model=GenerateQuestionResponse)
async def generate_question(
    context: InterviewContext, ai_service: AIServiceBase = Depends(get_ai_service)
):
    """
    Receives the interview context and generates the next question.
    """
    question_text = ai_service.generate_interview_question(context.model_dump())
    return GenerateQuestionResponse(
        next_question=question_text, is_interview_finished=False
    )


@router.post("/evaluate", response_model=EvaluateAnswerResponse)
async def evaluate_answer(
    request: EvaluateAnswerRequest, ai_service: AIServiceBase = Depends(get_ai_service)
):
    """
    Evaluates a single TEXT answer to extract behavioral metrics.
    """
    evaluation = ai_service.evaluate_answer(request.question, request.candidate_answer)
    return EvaluateAnswerResponse(observed_behaviors=evaluation)
