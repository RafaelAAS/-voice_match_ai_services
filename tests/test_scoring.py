from app.services.scoring_service import ScoringService


class TestScoringService:

    def test_calculate_behavioral_score_normal(self):
        """Testa o cálculo padrão com pesos diferentes."""
        behaviors = {
            "proactivity": {"desired_score": 9.0, "observed_score": 8.0},
            "teamwork": {"desired_score": 5.0, "observed_score": 5.0},
        }
        score = ScoringService.calculate_behavioral_score(behaviors)
        assert score == 9.36

    def test_calculate_behavioral_score_zero_weight(self):
        """Testa a proteção contra divisão por zero quando o desired_score é 0."""
        behaviors = {"ignored_trait": {"desired_score": 0.0, "observed_score": 5.0}}
        score = ScoringService.calculate_behavioral_score(behaviors)
        assert score == 5.0

    def test_calculate_behavioral_score_empty(self):
        """Testa o retorno quando o dicionário de características está vazio."""
        score = ScoringService.calculate_behavioral_score({})
        assert score == 0.0

    def test_calculate_overall_score(self):
        """Testa se a fórmula final aplica as porcentagens exigidas pelas
        regras de negócio."""
        score = ScoringService.calculate_overall_score(
            technical_score=8.0,
            experience_score=7.0,
            behavioral_score=9.0,
            communication_score=8.5,
            ideal_candidate_score=9.0,
        )
        assert score == 8.3
