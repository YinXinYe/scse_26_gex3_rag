### COMPLETE THE CODE ###
import os

# Route HuggingFace model downloads through a reachable mirror
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from rag import rag_answer


class Agent:
    """University IT support agent that answers questions with RAG."""

    def answer(self, question):
        """Return the agent's answer for a user question."""
        return rag_answer(question)


if __name__ == "__main__":
    agent = Agent()
    q = "What exact phone number should I call for the University IT Service Desk?"
    print("QUESTION:", q)
    print("ANSWER:", agent.answer(q))
