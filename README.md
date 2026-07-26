# VoiceMatch AI Services

Microsserviço responsável pelo "cérebro" de inteligência artificial do ecossistema VoiceMatch.

Este serviço atua como um recrutador técnico automatizado. Ele recebe dados de entrevistas (áudio ou texto), processa o contexto da vaga e utiliza LLMs para extrair métricas comportamentais (Proatividade, Resolução de Problemas, Trabalho em Equipe), transcrever respostas e gerar as próximas perguntas da entrevista de forma dinâmica.

**Atualização Recente:** A engine de Inteligência Artificial foi migrada para a Groq API, aproveitando a altíssima velocidade de inferência (LPU) para garantir respostas em tempo real durante as entrevistas.

---

## Tecnologias Utilizadas

* **Linguagem:** Python 3.10+
* **Framework Web:** FastAPI (Alta performance e documentação automática)
* **Inteligência Artificial:** Groq API (LLM / Processamento de Linguagem Natural)
* **Containerização:** Docker & Docker Compose
* **Testes e Qualidade:** Pytest & pytest-cov

---

## Variáveis de Ambiente (.env)

Antes de executar o projeto, crie um arquivo `.env` na raiz do repositório baseado no `.env.example` (se houver) e preencha com as suas configurações:

```env
ENVIRONMENT=development
PORT=8001
MOCK_AI=false

# Chave de API da Groq (Obrigatória se MOCK_AI=false)
GROQ_API_KEY=sua_chave_groq_aqui

# Configurações de Entrevista
MIN_INTERVIEW_QUESTIONS=8
MAX_INTERVIEW_QUESTIONS=12
```

---

## Execução via Docker (Recomendado)

O projeto já está totalmente configurado para rodar em containers, facilitando a integração com o Gateway e outros serviços do VoiceMatch (compartilhamento de volumes de áudio, etc).

1. Construa e suba os containers em segundo plano:
   ```bash
   docker-compose up -d --build
   ```
2. Para acompanhar os logs da IA em tempo real:
   ```bash
   docker-compose logs -f
   ```
3. Para derrubar os containers:
   ```bash
   docker-compose down
   ```

---

## Execução Local (Ambiente de Desenvolvimento)

Caso queira rodar a aplicação diretamente na sua máquina para debugar ou criar novas rotas:

1. **Crie e ative o ambiente virtual:**
   ```bash
   # Windows
   python -m venv .venv
   .venv\Scripts\activate

   # Linux/Mac
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. **Instale as dependências:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Inicie o servidor FastAPI:**
   ```bash
   uvicorn app.main:app --port 8001 --reload
   ```

---

## Documentação da API (Swagger UI)

O FastAPI gera automaticamente a documentação interativa das rotas. Com o servidor rodando (via Docker ou local), acesse o navegador:

**[http://localhost:8001/docs](http://localhost:8001/docs)**

Lá você poderá testar todas as rotas ativas, como:
* POST /ai/evaluate-audio (Processamento de áudio físico mapeado)
* POST /ai/evaluate (Avaliação baseada em texto)
* POST /ai/generate-question (Geração de próxima pergunta contextual)

---

## Testes Unitários e Cobertura

O projeto utiliza pytest para garantir a integridade das funções e do processamento da IA.

Para rodar todos os testes e gerar o relatório de cobertura de código (Coverage), execute no seu terminal (dentro do ambiente virtual):

```bash
pytest --cov=app tests/
```

Para gerar um relatório HTML mais detalhado da cobertura:
```bash
pytest --cov=app --cov-report=html tests/
```
*(Isso criará uma pasta `htmlcov`. Basta abrir o arquivo `index.html` no seu navegador).*
