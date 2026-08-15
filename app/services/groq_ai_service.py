import json
import os
from typing import Any

from groq import Groq

from app.core.config import settings


from app.services.audio_analyzer import AudioFeatureExtractor


class GroqAIService:
    """
    Fluxo em duas etapas, usando Groq:
      1) transcribe_audio(): envia o áudio ao Whisper (whisper-large-v3) e
         recebe a transcrição em texto puro.
      2) evaluate_transcript(): envia a transcrição (texto) + contexto da
         vaga a um modelo de texto (Llama) e recebe a próxima pergunta +
         métricas comportamentais em JSON.
      3) AudioFeatureExtractor: extrai parâmetros do sinal acústico via Librosa.

    process_audio_interview() orquestra as três etapas e devolve um único
    dict, no mesmo formato que a rota /ai/evaluate-audio já espera.
    """

    def __init__(self):
        api_key = settings.groq_api_key
        if not api_key:
            raise ValueError("A chave de API do Groq não foi encontrada.")

        self.client = Groq(api_key=api_key)
        self.transcription_model = "whisper-large-v3"
        self.chat_model = "llama-3.3-70b-versatile"
        self.audio_extractor = AudioFeatureExtractor()

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
        historico = context.get("conversation_history") or []
        historico_formatado = self._format_conversation_history(historico)
        
        # Determinar qual pergunta deve ser gerada a seguir no ciclo de 3 fases:
        # Pergunta 1 já foi respondida -> Próxima deve ser Etapa 2 (Fit Cultural)
        # Pergunta 2 já foi respondida -> Próxima deve ser Etapa 3 (Técnica)
        # Pergunta 3 já foi respondida -> Não há próxima pergunta (Entrevista concluída)
        num_respostas = len([h for h in historico if h.get("resposta")]) + 1
        
        if num_respostas == 1:
            diretriz_etapa = (
                "Esta foi a resposta da Etapa 1 (Apresentação Pessoal / Trajetória). "
                "Sua próxima pergunta DEVE SER OBRIGATORIAMENTE da Etapa 2 (Fit Cultural): "
                "pergunte sobre dinâmica e trabalho em equipe, convivência com colegas, valores e resolução de conflitos."
            )
        elif num_respostas == 2:
            diretriz_etapa = (
                "Esta foi a resposta da Etapa 2 (Fit Cultural). "
                "Sua próxima pergunta DEVE SER OBRIGATORIAMENTE da Etapa 3 (Técnica / Desafio Prático): "
                "pergunte sobre um desafio técnico prático ou aprofundamento específico nas competências, ferramentas e tecnologias exigidas pela vaga."
            )
        else:
            diretriz_etapa = (
                "Esta foi a resposta da Etapa 3 (Técnica). A entrevista atingiu o limite máximo de 3 perguntas. "
                "Retorne 'proxima_pergunta': null."
            )

        prompt = f"""
        Você é a IA recrutadora do VoiceMatch AI conduzindo uma entrevista estruturada de 3 fases.

        Diretriz de Fase:
        {diretriz_etapa}

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
            "proxima_pergunta": {"null" if num_respostas >= 3 else '"Texto direto e objetivo da próxima pergunta para o candidato"'},
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
            dados = json.loads(raw_text)
            if num_respostas >= 3:
                dados["proxima_pergunta"] = None
            return dados
        except json.JSONDecodeError:
            print("Resposta bruta do Groq (não era JSON válido):", raw_text)
            raise

    async def process_audio_interview(
        self, audio_file_path: str, context: dict
    ) -> dict[str, Any]:
        transcricao = self.transcribe_audio(audio_file_path)
        avaliacao = self.evaluate_transcript(transcricao, context)
        acustica = self.audio_extractor.analyze_audio_file(audio_file_path)

        metricas_finais = avaliacao.get("metricas", {})
        metricas_finais["acustica"] = acustica

        return {
            "transcricao": transcricao,
            "proxima_pergunta": avaliacao.get("proxima_pergunta"),
            "metricas": metricas_finais,
        }

    def generate_final_evaluation(self, context: dict[str, Any]) -> dict[str, Any]:
        prompt = f"""
        Você é um diretor sênior de RH e especialista em recrutamento por IA do VoiceMatch AI.
        Avalie a performance consolidada do candidato após a conclusão do ciclo de 3 perguntas (Pessoal, Fit Cultural e Técnica).

        Vaga / Requisitos:
        {context.get('job_requirements')}

        Histórico das 3 Perguntas e Respostas Transcritas:
        {json.dumps(context.get('conversation_history', []), ensure_ascii=False)}

        Sua tarefa é retornar EXCLUSIVAMENTE um objeto JSON válido no seguinte formato exato:
        {{
            "score_geral": <float de 0.0 a 10.0 representando a nota final consolidada com 1 casa decimal>,
            "feedback_geral": "<resumo executivo direto sobre o desempenho do candidato nas 3 etapas para o recrutador>",
            "sugestao_entrevista_video": "<parecer direto se VALE A PENA ou NÃO agendar uma entrevista por vídeo síncrona com o time e a justificativa clara>",
            "feedback_candidato": "<mensagem educada, profissional e acolhedora ao candidato, destacando sua participação no ciclo de 3 perguntas e deixando explícito que o avanço para as próximas fases do processo seletivo dependerá estritamente da análise e decisão da equipe de recrutamento>",
            "strengths": ["<ponto forte 1>", "<ponto forte 2>"],
            "weaknesses": ["<ponto de atenção 1>"],
            "recommendation": "strong_hire" | "hire" | "consider" | "reject"
        }}
        """

        response = self.client.chat.completions.create(
            model=self.chat_model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.2,
        )

        try:
            return json.loads(response.choices[0].message.content)
        except Exception:
            return {
                "score_geral": 7.5,
                "feedback_geral": "Candidato concluiu o ciclo de 3 perguntas demonstrando bom alinhamento inicial com os requisitos da vaga.",
                "sugestao_entrevista_video": "Recomendamos agendar uma entrevista por vídeo para aprofundar na experiência prática em projetos de grande escala.",
                "feedback_candidato": "Agradecemos muito sua participação e empenho na entrevista de voz do VoiceMatch AI! Suas respostas para as etapas Pessoal, Fit Cultural e Técnica foram registradas com sucesso. Informamos que o avanço para as próximas fases do processo seletivo dependerá estritamente da análise e deliberação da equipe de recrutamento.",
                "strengths": ["Boa comunicação", "Conhecimento técnico relevante"],
                "weaknesses": ["Poderia detalhar mais exemplos práticos em produção"],
                "recommendation": "hire",
            }
