from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


class TestAPI:

    def test_health_check(self):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}

    def test_transcribe_audio(self):
        file_content = b"fake audio content bytes"
        files = {"file": ("test_audio.mp3", file_content, "audio/mpeg")}

        response = client.post("/ai/transcribe", files=files)

        assert response.status_code == 200
        data = response.json()
        assert "transcription" in data
        assert isinstance(data["transcription"], str)

    def test_generate_question(self):
        payload = {
            "job_requirements": "Requires strong Python and Docker skills.",
            "behavioral_profile": {"proactivity": 9, "teamwork": 8},
            "candidate_resume": "Backend engineer with 3 years of experience.",
            "conversation_history": [
                {"sender": "ai", "content": "Welcome to the interview."}
            ],
        }

        response = client.post("/ai/generate-question", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert "next_question" in data
        assert "is_interview_finished" in data
        assert isinstance(data["is_interview_finished"], bool)

    def test_evaluate_answer(self):
        payload = {
            "question": "Can you give an example of a complex problem you solved?",
            "candidate_answer": "I optimized a database query that was timing out.",
        }

        response = client.post("/ai/evaluate", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert "observed_behaviors" in data
        assert isinstance(data["observed_behaviors"], dict)
