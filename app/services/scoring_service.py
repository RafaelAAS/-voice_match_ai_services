class ScoringService:

    @staticmethod
    def calculate_behavioral_score(behaviors: dict[str, dict[str, float]]) -> float:
        """
        Calculates the behavioral score based on the proximity between
        desired and observed scores. Traits with a higher 'desired_score'
        have a heavier weight in the final average.
        """
        total_weight = 0.0
        weighted_sum = 0.0

        for data in behaviors.values():
            desired = data.get("desired_score", 0.0)
            observed = data.get("observed_score", 0.0)

            trait_score = 10.0 - abs(desired - observed)

            weight = desired if desired > 0 else 1.0

            weighted_sum += trait_score * weight
            total_weight += weight

        if total_weight == 0:
            return 0.0

        return round(weighted_sum / total_weight, 2)

    @staticmethod
    def calculate_overall_score(
        technical_score: float,
        experience_score: float,
        behavioral_score: float,
        communication_score: float,
        ideal_candidate_score: float,
    ) -> float:
        """
        Calculates the final candidate score based on predefined weights,
        rounded to one decimal place.
        """
        overall_score = (
            (technical_score * 0.25)
            + (experience_score * 0.20)
            + (behavioral_score * 0.25)
            + (communication_score * 0.10)
            + (ideal_candidate_score * 0.20)
        )

        return round(overall_score, 1)
