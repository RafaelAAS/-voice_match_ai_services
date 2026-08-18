from typing import Any

from pydantic import BaseModel, Field


class InterviewContext(BaseModel):
    job_requirements: str | None = ""
    behavioral_profile: dict[str, Any] | None = None
    candidate_resume: str | None = ""
    conversation_history: list[dict[str, Any]] | None = []


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
    context: InterviewContext | None = None


class EvaluateAudioResponse(BaseModel):
    transcricao: str | None = Field(
        None, description="Transcrição integral da fala do candidato."
    )
    proxima_pergunta: str | None = Field(
        None, description="Próxima pergunta a ser feita ao candidato."
    )
    metricas: dict[str, Any] | None = Field(
        None, description="Métricas comportamentais e prosódicas extraídas."
    )
