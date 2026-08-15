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
            "score_geral": 8.5,
            "feedback_geral": "O candidato concluiu com êxito as 3 etapas da entrevista (Pessoal, Fit Cultural e Técnica), demonstrando sólida base técnica e ótima comunicação.",
            "sugestao_entrevista_video": "Recomendamos fortemente o agendamento de uma entrevista por vídeo para alinhamento final de expectativas e proposta.",
            "feedback_candidato": "Agradecemos imensamente sua dedicação e respostas nas 3 etapas da nossa entrevista de voz! Suas colocações foram registradas. Lembramos que o avanço para as próximas fases do processo seletivo dependerá estritamente da avaliação e deliberação da equipe de recrutamento.",
            "strengths": ["Comunicação clara e articulada", "Boa resolução de conflitos em equipe", "Domínio das tecnologias exigidas"],
            "weaknesses": ["Pode aprofundar em exemplos práticos de arquitetura sob alta escala"],
            "recommendation": "strong_hire",
        }

    async def process_audio_interview(
        self, audio_file_path: str, context: dict
    ) -> dict[str, Any]:
        historico = context.get("conversation_history") or []
        num_respostas = len([h for h in historico if h.get("resposta")]) + 1

        if num_respostas == 1:
            proxima = "Muito obrigado pela sua apresentação! Entrando na nossa etapa de Fit Cultural: como você lida com divergências de opiniões e trabalho em equipe em momentos de alta pressão?"
        elif num_respostas == 2:
            proxima = "Excelente reflexão. Agora para nossa etapa Técnica: conte-me sobre um desafio técnico complexo que você enfrentou na sua stack principal e como foi a sua abordagem para resolvê-lo."
        else:
            proxima = None

        return {
            "transcricao": "Esta é uma transcrição demonstrativa da sua resposta por áudio gravada pelo microfone.",
            "proxima_pergunta": proxima,
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
