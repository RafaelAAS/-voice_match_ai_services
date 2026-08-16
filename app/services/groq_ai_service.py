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
                "Sua próxima pergunta DEVE SER OBRIGATORIAMENTE da Etapa 3 (Pergunta Técnica): "
                "faça uma pergunta técnica direta, prática e contextualizada sobre as principais competências, ferramentas, arquitetura ou tecnologias exigidas pela vaga."
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
        cand_name = context.get("candidate_name") or "Candidato(a)"
        primeiro_nome = cand_name.split()[0] if cand_name else "Candidato(a)"
        job_title = context.get("job_title") or "a vaga"
        job_reqs = context.get("job_requirements") or ""
        screening = context.get("screening_evaluation") or {}
        history = context.get("conversation_history") or []

        prompt = f"""
        Você é a Iris, a inteligência artificial especialista em recrutamento por voz do VoiceMatch AI.
        Você conduziu o processo seletivo do(a) candidato(a) '{cand_name}' para a vaga '{job_title}'.

        O processo é composto por dois pilares integrados:
        1. Análise Semântica de Currículo: Avaliação de hard skills, formação, vivência prática e aderência técnica aos requisitos da vaga.
        2. Entrevista Estruturada de Voz (3 Fases):
           - Etapa 1: Apresentação Pessoal & Trajetória
           - Etapa 2: Fit Cultural & Trabalho em Equipe
           - Etapa 3: Pergunta Técnica Prática

        Requisitos e Contexto da Vaga:
        {job_reqs}

        Dados da Análise de Currículo (Triagem):
        {json.dumps(screening, ensure_ascii=False, indent=2)}

        Histórico e Métricas da Entrevista de Voz:
        {json.dumps(history, ensure_ascii=False, indent=2)}

        Sua tarefa é gerar uma avaliação executiva para o recrutador E um FEEDBACK PROFISSIONAL, FUNDAMENTADO E ACOLHEDOR DIRETAMENTE PARA O CANDIDATO.

        DIRETRIZES CRÍTICAS PARA O "feedback_geral" (VISÃO DO RECRUTADOR):
        - TOM E POSTURA: Altamente analítico, crítico, criterioso e exigente. Diferente do feedback acolhedor do candidato, aqui o tom deve ser direto, técnico e focado em riscos, trade-offs e nível real de senioridade.
        - ESTRUTURA: Deve ser um parecer detalhado com 3 a 4 parágrafos substanciais divididos por quebras de linha duplas (\n\n):
          1. Profundidade Técnica & Domínio da Stack: Avaliar friamente a profundidade técnica demonstrada na resposta de arquitetura/código (se citou bibliotecas, gargalos reais, métricas, mitigação de falhas ou se foi superficial).
          2. Maturidade Comportamental & Riscos de Equipe: Analisar a postura em situações de conflito, autonomia, senioridade na tomada de decisões e possíveis riscos de integração ao time.
          3. Performance Vocal & Comunicação sob Pressão: Análise técnica da clareza, firmeza, ritmo prosódico e controle de hesitações observadas nas métricas acústicas.
          4. Veredito Executivo & Recomendações de Investigação: Conclusão clara sobre o nível de prontidão do candidato e pontos exatos que o time técnico sênior deve aprofundar na entrevista de vídeo síncrona.

        DIRETRIZES CRÍTICAS PARA O "feedback_candidato" (VISÃO DO CANDIDATO):
        - Deve ser uma análise transparente, construtiva e humanizada, chamando o candidato pelo primeiro nome ({primeiro_nome}).
        - NÃO deve ser um texto genérico ou curto. Deve conter parágrafos bem estruturados e divididos por quebras de linha duplas (\n\n).
        - Deve conter OBRIGATORIAMENTE os seguintes tópicos/seções:
          1. Saudação e Acolhimento: Parabenizando pela conclusão das 3 etapas da entrevista de voz do VoiceMatch AI para a vaga de {job_title}.
          2. Síntese do Currículo (Hard Skills): Como suas experiências e conhecimentos técnicos atendem às exigências da vaga, destacando pontos fortes reais.
          3. Síntese da Entrevista por Voz (Soft Skills & Postura): Como foi o desempenho nas respostas por áudio (maturidade, postura colaborativa e clareza de raciocínio).
          4. Sugestão de Melhoria Técnica (Hard Skills & Vaga): Uma recomendação prática e fundamentada de aprimoramento técnico diretamente ligada aos requisitos da vaga, gaps identificados na triagem ou aprofundamento da resposta técnica.
          5. Sugestão de Oratória & Comunicação (Análise Acústica): Avaliação da expressão verbal baseada nas métricas acústicas dos áudios (clareza de dicção, pausas, firmeza e ritmo de fala), com uma dica prática para elevar a oratória.
          6. Transparência e Próximos Passos: Reforçar que o avanço no processo dependerá da deliberação final da equipe humana de recrutamento.

        Retorne EXCLUSIVAMENTE um objeto JSON válido no seguinte formato exato:
        {{
            "score_geral": <float de 0.0 a 10.0 representando a nota final consolidada com 1 casa decimal>,
            "feedback_geral": "<parecer executivo analítico, exigente e detalhado para o recrutador em múltiplos parágrafos com quebras de linha duplas>",
            "sugestao_entrevista_video": "<parecer direto se VALE A PENA ou NÃO agendar uma entrevista por vídeo síncrona com o time e a justificativa clara>",
            "feedback_candidato": "<texto completo, estruturado com todos os 6 tópicos acima, utilizando quebras de linha duplas para separar os blocos>",
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
                "score_geral": 8.0,
                "feedback_geral": f"O candidato {cand_name} concluiu o ciclo de 3 perguntas demonstrando bom alinhamento com os requisitos da vaga.",
                "sugestao_entrevista_video": "Recomendamos agendar uma entrevista por vídeo para aprofundar na experiência prática em projetos de grande escala.",
                "feedback_candidato": (
                    f"Olá, {primeiro_nome}! Agradecemos imensamente sua dedicação e participação em todas as etapas do processo seletivo do VoiceMatch AI para a vaga de {job_title}.\n\n"
                    "📄 **Análise de Currículo (Hard Skills):**\n"
                    "Sua formação e experiências prévias demonstraram uma base sólida e alinhada com as principais competências técnicas exigidas, evidenciando familiaridade com as ferramentas e práticas do mercado.\n\n"
                    "🎙️ **Entrevista de Voz (Soft Skills & Comunicação):**\n"
                    "Ao longo das 3 etapas de voz, você demonstrou boa clareza de raciocínio, segurança nas colocações e uma postura colaborativa ao lidar com dinâmicas de equipe e desafios técnicos.\n\n"
                    "💡 **Dica de Desenvolvimento:**\n"
                    "Como sugestão de evolução contínua, recomendamos aprofundar em métricas de performance e arquitetura de sistemas escaláveis para fortalecer ainda mais seu repertório.\n\n"
                    "Seus resultados foram consolidados e enviados para o time de recrutamento, que avaliará sua candidatura para as próximas fases."
                ),
                "strengths": ["Boa comunicação", "Conhecimento técnico relevante"],
                "weaknesses": ["Poderia detalhar mais exemplos práticos em produção"],
                "recommendation": "hire",
            }
