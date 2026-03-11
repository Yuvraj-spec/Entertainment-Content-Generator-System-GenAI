from src.retrieval import hybrid_search
from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="tinyllama")

def generate_scene(user_prompt):

    docs = hybrid_search(user_prompt, k=3)
    context = "\n".join([doc.page_content for doc in docs[:2]])

    prompt = f"""
You are a screenplay writer.

Using the context below, write a short original scene.

Context:
{context}

User prompt:
{user_prompt}

Write a 10–15 line screenplay scene.
"""

    response = llm.invoke(
        prompt,
        max_new_tokens=200,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2
    )

    return response