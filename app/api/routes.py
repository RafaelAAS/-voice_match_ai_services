from fastapi import APIRouter, Depends, File, UploadFile

from app.schemas.ai_schemas import (
    EvaluateAnswerRequest,
    EvaluateAnswerResponse,
    GenerateQuestionResponse,
    InterviewContext,
)
from app.services.ai_service_base import AIServiceBase
from app.services.mock_ai_service import get_ai_service

router = APIRouter(prefix="/ai", tags=["AI Operations"])


@router.post("/transcribe", response_model=dict)
async def transcribe_audio(
    file: UploadFile = File(...), ai_service: AIServiceBase = Depends(get_ai_service)
):
    """
    Receives an audio file from the Gateway and returns the transcribed text.
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
    Evaluates a single answer to extract behavioral metrics.
    """
    evaluation = ai_service.evaluate_answer(request.question, request.candidate_answer)

    return EvaluateAnswerResponse(observed_behaviors=evaluation)
