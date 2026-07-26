from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    environment: str = "development"
    port: int = 8001
    mock_ai: bool = False

    openai_api_key: str = ""
    elevenlabs_api_key: str = ""
    google_api_key: str = ""
    groq_api_key: str = ""

    min_interview_questions: int = 8
    max_interview_questions: int = 12

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


settings = Settings()
