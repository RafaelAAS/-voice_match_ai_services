from abc import ABC, abstractmethod
from typing import Any


class AIServiceBase(ABC):

    @abstractmethod
    def transcribe_audio(self, audio_file_path: str) -> str:
        """Receives the audio file path and returns the transcribed text."""
        pass

    @abstractmethod
    def generate_interview_question(self, context: dict[str, Any]) -> str:
        """Analyzes the context and generates the next interview question."""
        pass

    @abstractmethod
    def generate_audio(self, text: str) -> str:
        """Converts text to audio via TTS and returns the URL."""
        pass

    @abstractmethod
    def evaluate_answer(self, question: str, answer: str) -> dict[str, Any]:
        """Evaluates a single answer to extract observed behaviors."""
        pass

    @abstractmethod
    def generate_final_evaluation(self, context: dict[str, Any]) -> dict[str, Any]:
        """Generates the final structured summary after the interview ends."""
        pass
