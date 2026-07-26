from typing import Any

from pydantic import BaseModel


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
