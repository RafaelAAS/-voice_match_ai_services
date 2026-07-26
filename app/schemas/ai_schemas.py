from typing import Any

from pydantic import BaseModel, Field


class InterviewContext(BaseModel):
    job_requirements: str
    behavioral_profile: dict[str, int]
    candidate_resume: str
    conversation_history: list[dict[str, str]]


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
    context: InterviewContext


class InterviewMetrics(BaseModel):
    proatividade: int = Field(..., ge=0, le=10)
    resolucao_de_problemas: int = Field(..., ge=0, le=10)
    trabalho_em_equipe: int = Field(..., ge=0, le=10)


class EvaluateAudioResponse(BaseModel):
    transcricao: str = Field(
        ..., description="Transcrição integral da fala do candidato."
    )
    proxima_pergunta: str = Field(
        ..., description="Próxima pergunta a ser feita ao candidato."
    )
    metricas: InterviewMetrics
