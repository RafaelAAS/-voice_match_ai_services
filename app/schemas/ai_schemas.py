from typing import Any, Optional

from pydantic import BaseModel, Field


class InterviewContext(BaseModel):
    job_requirements: Optional[str] = ""
    behavioral_profile: Optional[dict[str, Any]] = None
    candidate_resume: Optional[str] = ""
    conversation_history: Optional[list[dict[str, Any]]] = []


class GenerateQuestionResponse(BaseModel):
    next_question: str
    is_interview_finished: bool


class EvaluateAnswerRequest(BaseModel):
    question: str
    candidate_answer: str


class EvaluateAnswerResponse(BaseModel):
    observed_behaviors: dict[str, Any]


class FinalEvaluationResponse(BaseModel):
    summary: str
    strengths: list[str]
    weaknesses: list[str]
    improvements: list[str]
    recommendation: str


class EvaluateAudioRequest(BaseModel):
    audio_path: str = Field(
        ..., description="Caminho do áudio no volume compartilhado."
    )
    context: Optional[InterviewContext] = None


class EvaluateAudioResponse(BaseModel):
    transcricao: Optional[str] = Field(
        None, description="Transcrição integral da fala do candidato."
    )
    proxima_pergunta: Optional[str] = Field(
        None, description="Próxima pergunta a ser feita ao candidato."
    )
    metricas: Optional[dict[str, Any]] = Field(
        None, description="Métricas comportamentais e prosódicas extraídas."
    )
