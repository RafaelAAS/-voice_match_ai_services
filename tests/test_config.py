from app.core.config import settings


class TestConfig:
    def test_settings_loaded(self):
        """Garante que as variáveis de ambiente base foram carregadas."""
        assert isinstance(settings.environment, str)
        assert isinstance(settings.port, int)
        assert isinstance(settings.mock_ai, bool)
        assert settings.min_interview_questions == 8
        assert settings.max_interview_questions == 12
