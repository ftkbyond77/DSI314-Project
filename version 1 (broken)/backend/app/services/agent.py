from .langchain_client import get_llm
from .pinecone_client import upsert_embeddings, query_embeddings
from .memory import MemoryManager

class Agent:
    def __init__(self, user_id):
        self.user_id = user_id
        self.llm = get_llm()
        self.memory = MemoryManager(user_id)

    def plan_study(self, files):
        vectors = []
        for f in files:
            # GPT-5 Create Embedding
            vector = self.llm.embed(f.file_content)  
            upsert_embeddings(f"file-{f.id}", vector)
            vectors.append(vector)

        # Query Pinecone
        context = [query_embeddings(v) for v in vectors]

        # Generate plan
        prompt = f"""
        User uploaded {len(files)} files.
        Context: {context}
        Generate a step-by-step study plan with priority ranking.
        """
        plan = self.llm(prompt)
        self.memory.save_plan(plan)
        return plan

    def generate_quiz(self, file):
        prompt = f"Generate 5 quiz questions from the following content:\n{file.file_content}"
        quiz = self.llm(prompt)
        return quiz

    def feedback_loop(self, progress, quiz_scores):
        # check progress and quiz_score then score plan
        prompt = f"User progress: {progress}, Quiz scores: {quiz_scores}. Suggest adjustments to study plan."
        adjusted_plan = self.llm(prompt)
        self.memory.update_plan(adjusted_plan)
        return adjusted_plan
