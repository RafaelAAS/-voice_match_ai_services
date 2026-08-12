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

    async def process_audio_interview(
        self, audio_file_path: str, context: dict
    ) -> dict[str, Any]:
        return {
            "transcricao": "Esta é uma transcrição demonstrativa da sua resposta por áudio.",
            "proxima_pergunta": "Como você lida com prazos apertados e priorização de tarefas em projetos complexos?",
            "metricas": {
                "proatividade": 8,
                "resolucao_de_problemas": 8,
                "trabalho_em_equipe": 9,
                "acustica": {
                    "pitch_f0_hz": 185.4,
                    "rms_energy": 0.042,
                    "speech_rate_wpm": 145.0,
                    "pause_hesitation_sec": 0.8,
                    "soft_skills_scores": {
                        "oratoria": 8.5,
                        "firmeza_vocal": 8.0,
                        "controle_estresse": 8.8,
                        "entusiasmo": 8.2,
                    },
                },
            },
        }


def get_ai_service() -> AIServiceBase:
    """
    Factory function to inject the Mock service into the FastAPI routes.
    In the future, this will check an environment variable to return RealAIService.
    """
    return MockAIService()
