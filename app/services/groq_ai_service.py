import json
import os
from typing import Any

from groq import Groq

from app.core.config import settings


class GroqAIService:
    """
    Fluxo em duas etapas, usando Groq:
      1) transcribe_audio(): envia o áudio ao Whisper (whisper-large-v3) e
         recebe a transcrição em texto puro.
      2) evaluate_transcript(): envia a transcrição (texto) + contexto da
         vaga a um modelo de texto (Llama) e recebe a próxima pergunta +
         métricas comportamentais em JSON.

    process_audio_interview() orquestra as duas etapas e devolve um único
    dict, no mesmo formato que a rota /ai/evaluate-audio já espera.
    """

    def __init__(self):
        api_key = settings.groq_api_key
        if not api_key:
            raise ValueError("A chave de API do Groq não foi encontrada.")

        self.client = Groq(api_key=api_key)
        self.transcription_model = "whisper-large-v3"
        self.chat_model = "llama-3.3-70b-versatile"

    def transcribe_audio(self, audio_file_path: str) -> str:
        if not os.path.exists(audio_file_path):
            raise FileNotFoundError(
                f"Arquivo de áudio não encontrado no caminho: {audio_file_path}"
            )

        with open(audio_file_path, "rb") as audio_file:
            transcription = self.client.audio.transcriptions.create(
                file=audio_file,
                model=self.transcription_model,
                language="pt",
                response_format="text",
            )

        transcricao = (
            transcription if isinstance(transcription, str) else transcription.text
        )
        transcricao = transcricao.strip()

        if not transcricao:
            raise ValueError("O Whisper retornou uma transcrição vazia.")
        return transcricao

    @staticmethod
    def _format_behavioral_profile(behavioral_profile: dict[str, int] | None) -> str:
        if not behavioral_profile:
            return "Não especificado."
        linhas = [
            f"- {competencia}: nota {nota}/10"
            for competencia, nota in behavioral_profile.items()
        ]
        return "\n".join(linhas)

    @staticmethod
    def _format_conversation_history(
        conversation_history: list[dict[str, str]] | None
    ) -> str:
        if not conversation_history:
            return "Esta é a primeira resposta do candidato na entrevista."

        linhas = []
        for i, turno in enumerate(conversation_history, start=1):
            partes = [f"{chave}: {valor}" for chave, valor in turno.items()]
            linhas.append(f"{i}. " + " | ".join(partes))
        return "\n".join(linhas)

    def evaluate_transcript(self, transcricao: str, context: dict) -> dict[str, Any]:
        perfil_formatado = self._format_behavioral_profile(
            context.get("behavioral_profile")
        )
        historico_formatado = self._format_conversation_history(
            context.get("conversation_history")
        )

        prompt = f"""
        Você é um recrutador técnico avaliando um candidato.

        Requisitos da Vaga:
        {context.get('job_requirements')}

        Perfil Comportamental Desejado:
        {perfil_formatado}

        Histórico da Conversa até agora:
        {historico_formatado}

        O candidato respondeu agora (transcrição da fala dele):
        \"\"\"{transcricao}\"\"\"

        Devolva EXATAMENTE um JSON, sem blocos de formatação markdown, com:
        {{
            "proxima_pergunta": "Sua próxima pergunta como entrevistador
            para dar andamento à entrevista, focada nos requisitos da vaga.",
            "metricas": {{
                "proatividade": <nota de 0 a 10 baseada na resposta>,
                "resolucao_de_problemas": <nota de 0 a 10 baseada na resposta>,
                "trabalho_em_equipe": <nota de 0 a 10 baseada na resposta>
            }}
        }}
        """

        response = self.client.chat.completions.create(
            model=self.chat_model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )

        raw_text = response.choices[0].message.content
        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            print("Resposta bruta do Groq (não era JSON válido):", raw_text)
            raise

    async def process_audio_interview(
        self, audio_file_path: str, context: dict
    ) -> dict[str, Any]:
        transcricao = self.transcribe_audio(audio_file_path)
        avaliacao = self.evaluate_transcript(transcricao, context)

        return {
            "transcricao": transcricao,
            "proxima_pergunta": avaliacao.get("proxima_pergunta"),
            "metricas": avaliacao.get("metricas"),
        }
