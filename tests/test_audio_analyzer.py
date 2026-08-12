import os
import pytest
from app.services.audio_analyzer import AudioFeatureExtractor


def test_audio_analyzer_fallback_when_file_not_found():
    extractor = AudioFeatureExtractor()
    result = extractor.analyze_audio_file("caminho_inexistente_audio.wav")

    assert "soft_skills_acusticas" in result
    assert "oratoria_e_clareza" in result["soft_skills_acusticas"]
    assert "firmeza_e_confianca" in result["soft_skills_acusticas"]
    assert "controle_de_estresse" in result["soft_skills_acusticas"]
    assert "entusiasmo_e_engajamento" in result["soft_skills_acusticas"]
    assert "parecer_acustico" in result


def test_audio_analyzer_metrics_structure():
    extractor = AudioFeatureExtractor()
    result = extractor._fallback_features(reason="Teste unitario")

    assert result["duracao_total_segundos"] > 0
    assert isinstance(result["soft_skills_acusticas"]["oratoria_e_clareza"], (int, float))
    assert 0.0 <= result["soft_skills_acusticas"]["oratoria_e_clareza"] <= 10.0
