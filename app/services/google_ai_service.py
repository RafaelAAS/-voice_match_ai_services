import json
import os
from typing import Any

from google import genai
from google.genai import types

from app.core.config import settings


class GoogleAIService:
    """
    Fluxo em duas etapas:
      1) transcribe_audio(): envia SÓ o áudio ao Gemini e recebe a
      transcrição em texto puro.
      2) evaluate_transcript(): envia a transcrição
      (texto) + contexto da vaga ao Gemini
         e recebe a próxima pergunta + métricas comportamentais em JSON.

    process_audio_interview() orquestra as duas etapas e devolve um único dict
    já pronto para o voicematch-back persistir no banco.
    """

    def __init__(self):
        api_key = settings.google_api_key
        if not api_key:
            raise ValueError("A chave de API do Google não foi encontrada.")

        self.client = genai.Client(api_key=api_key)
        self.model_name = "gemini-flash-latest"

    # ------------------------------------------------------------------
    # Etapa 1: transcrição pura (áudio -> texto)
    #
    # NOTA: usamos bytes inline (types.Part.from_bytes) em vez de
    # client.files.upload(). Chaves novas do AI Studio no formato "AQ."
    # (Auth key) estão retornando 401 ACCESS_TOKEN_TYPE_UNSUPPORTED
    # especificamente no FileService/upload, mesmo com generate_content
    # funcionando normalmente (bug conhecido e reportado no fórum oficial
    # do Google em 2026). Enviar os bytes direto no generate_content evita
    # depender do endpoint de upload que está com esse problema.
    # ------------------------------------------------------------------
    def transcribe_audio(self, audio_file_path: str) -> str:
        if not os.path.exists(audio_file_path):
            raise FileNotFoundError(
                f"Arquivo de áudio não encontrado no caminho: {audio_file_path}"
            )

        with open(audio_file_path, "rb") as f:
            audio_bytes = f.read()

        mime_type = (
            "audio/mpeg" if audio_file_path.lower().endswith(".mp3") else "audio/wav"
        )
        audio_part = types.Part.from_bytes(data=audio_bytes, mime_type=mime_type)

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[
                "Transcreva exatamente o que foi dito neste áudio, em português. "
                "Devolva APENAS o texto transcrito, sem comentários, sem markdown, "
                "sem aspas ao redor.",
                audio_part,
            ],
        )
        transcricao = (response.text or "").strip()
        if not transcricao:
            raise ValueError("O Gemini retornou uma transcrição vazia.")
        return transcricao

    # ------------------------------------------------------------------
    # Etapa 2: avaliação a partir do texto já transcrito (sem áudio)
    # ------------------------------------------------------------------
    def evaluate_transcript(self, transcricao: str, context: dict) -> dict[str, Any]:
        prompt = f"""
        Você é um recrutador técnico avaliando um candidato.
        Contexto da entrevista:
        - Requisitos da Vaga: {context.get('job_requirements')}
        - Histórico da Conversa: {context.get('conversation_history')}
        - Perfil Comportamental Desejado: {context.get('behavioral_profile')}

        O candidato respondeu (transcrição da fala dele):
        \"\"\"{transcricao}\"\"\"

        Devolva EXATAMENTE um JSON, sem blocos de formatação markdown, com:
        {{
            "proxima_pergunta": "Sua próxima pergunta como
            entrevistador para dar andamento à entrevista,
            focada nos requisitos da vaga.",
            "metricas": {{
                "proatividade": <nota de 0 a 10 baseada na resposta>,
                "resolucao_de_problemas": <nota de 0 a 10 baseada na resposta>,
                "trabalho_em_equipe": <nota de 0 a 10 baseada na resposta>
            }}
        }}
        """

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )

        raw_text = response.text
        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            print("Resposta bruta do Gemini (não era JSON válido):", raw_text)
            raise

    # ------------------------------------------------------------------
    # Orquestração: junta as duas etapas no formato que o back-end consome
    # ------------------------------------------------------------------
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
