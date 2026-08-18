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
        cand_name = context.get("candidate_name") or "Candidato(a)"
        primeiro_nome = cand_name.split()[0] if cand_name else "Candidato(a)"
        job_title = context.get("job_title") or "Desenvolvedor Full Stack"
        return {
            "score_geral": 8.5,
            "feedback_geral": (
                f"Avaliação Executiva de Seleção — Candidato: {cand_name} (Posição: {job_title})\n\n"
                "1. Profundidade Técnica & Resolução Prática:\n"
                "O candidato demonstrou raciocínio analítico consistente na pergunta técnica, estruturando o diagnóstico de performance em etapas lógicas (identificação de gargalos em chamadas externas e refatoração com bibliotecas otimizadas). Contudo, a abordagem técnica permaneceu concentrada no nível de aplicação; recomenda-se que a banca técnica avalie na entrevista síncrona o domínio sobre indexação em banco de dados, estratégias de cache distribuído e tolerância a falhas sob volumetrias superiores a 100k req/min.\n\n"
                "2. Maturidade Comportamental & Dinâmica de Equipe:\n"
                "Na etapa de Fit Cultural, evidenciou maturidade ao separar aspectos interpessoais de divergências de projeto, priorizando critérios objetivos (prazos, impacto e manutenibilidade). Mostra perfil colaborativo com baixa propensão a conflitos destrutivos, adequado para ambientes ágeis.\n\n"
                "3. Performance Vocal & Controle sob Pressão:\n"
                "A análise prosódica revelou dicção clara, bom controle de ansiedade e firmeza vocal estável. Não foram detectadas hesitações excessivas, o que indica segurança nas respostas formuladas.\n\n"
                "4. Parecer Conclusivo para o Recrutador:\n"
                "Apresenta forte aderência ao perfil. Recomendado para avanço imediato para entrevista técnica por vídeo com foco em arquitetura de microsserviços e testes sob estresse."
            ),
            "sugestao_entrevista_video": "Recomendamos fortemente o agendamento de uma entrevista por vídeo para alinhamento final de expectativas e proposta.",
            "feedback_candidato": (
                f"Olá, {primeiro_nome}! Parabéns pela conclusão do seu ciclo de 3 etapas no VoiceMatch AI para a vaga de {job_title}.\n\n"
                "📄 **Análise Semântica de Currículo:**\n"
                "Seu currículo demonstrou excelente compatibilidade com as hard skills requeridas para a vaga, evidenciando domínio prático em desenvolvimento de software, estrutura de bancos de dados e boas práticas de engenharia.\n\n"
                "🎙️ **Entrevista por Voz (Soft Skills & Postura):**\n"
                "Durante as respostas por áudio, você apresentou ótima clareza de exposição, segurança técnica e maturidade ao discutir dinâmicas de equipe e resolução de problemas sob pressão.\n\n"
                "🛠️ **Sugestão de Melhoria Técnica:**\n"
                "Para fortalecer ainda mais seu perfil em relação aos requisitos desta vaga, recomendamos aprofundar em frameworks modernos de frontend (como Tailwind CSS e Angular) e em práticas avançadas de testes automatizados e arquitetura de microsserviços.\n\n"
                "🔊 **Análise Acústica & Oratória Vocal:**\n"
                "A análise prosódica dos seus áudios detectou uma fala com excelente firmeza vocal, boa modulação de pitch e cadência fluida. Como sugestão para aprimorar ainda mais sua oratória, procure utilizar micropausas estratégicas entre a apresentação do problema e a solução técnica para reforçar ainda mais o impacto da sua mensagem.\n\n"
                "Seu parecer consolidado foi encaminhado ao time de recrutamento, que fará a deliberação dos próximos passos."
            ),
            "strengths": [
                "Comunicação clara e articulada",
                "Boa resolução de conflitos em equipe",
                "Domínio das tecnologias exigidas",
            ],
            "weaknesses": [
                "Pode aprofundar em exemplos práticos de arquitetura sob alta escala"
            ],
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
            proxima = "Excelente reflexão. Agora para nossa etapa de Pergunta Técnica: conte-me sobre como você estrutura e aplica na prática as principais tecnologias e padrões de arquitetura da sua stack no dia a dia."
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
