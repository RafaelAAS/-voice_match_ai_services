from typing import Any

from app.services.ai_service_base import AIServiceBase


class MockAIService(AIServiceBase):

    def transcribe_audio(self, audio_file_path: str) -> str:
        return (
            "Esta é uma transcrição simulada em que o candidato responde "
            "sobre suas experiências anteriores."
        )

    def generate_interview_question(self, context: dict[str, Any]) -> str:
        return (
            "Interessante! Você poderia me dar um exemplo prático de "
            "como utilizou suas habilidades para resolver um problema sob pressão?"
        )

    def generate_audio(self, text: str) -> str:
        return "https://mock-storage.supabase.co/candidate-audios/mock_audio_123.mp3"

    def evaluate_answer(self, question: str, answer: str) -> dict[str, Any]:
        return {
            "proactivity": {
                "desired_score": 9,
                "observed_score": 8,
                "confidence": 0.85,
            },
            "problem_solving": {
                "desired_score": 8,
                "observed_score": 7,
                "confidence": 0.90,
            },
        }

    def generate_final_evaluation(self, context: dict[str, Any]) -> dict[str, Any]:
        return {
            "summary": "O candidato demonstrou sólida base técnica e boa "
            "adequação ao perfil comportamental.",
            "strengths": ["Conhecimento em automação", "Trabalho em equipe"],
            "weaknesses": ["Comunicação sob forte pressão"],
            "improvements": ["Explorar mais práticas de documentação de código"],
            "recommendation": "good_match",
        }


def get_ai_service() -> AIServiceBase:
    """
    Factory function to inject the Mock service into the FastAPI routes.
    In the future, this will check an environment variable to return RealAIService.
    """
    return MockAIService()
