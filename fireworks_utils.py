import os
from fireworks import LLM

# Get Fireworks API key from environment variable or config
def get_fireworks_llm(model="accounts/fireworks/models/llama-v3p2-3b-instruct", temperature=0.2):
    api_key = os.getenv("FIREWORKS_API_KEY")
    if not api_key:
        raise ValueError("Fireworks API key not set. Please set FIREWORKS_API_KEY environment variable.")
    # Model options: Llama-3, Mistral, etc. See Fireworks docs for latest names.
    llm = LLM(
        model=model,
        temperature=temperature,
        deployment_type="auto"
    )
    return llm

def synthesize_answer_with_fireworks(question: str, context: str, model="accounts/fireworks/models/llama-v3p2-3b-instruct") -> str:
    """
    Use Fireworks LLM to synthesize a concise answer from retrieved context.
    """
    llm = get_fireworks_llm(model=model)
    # Prompt template (optimized for RAG)
    prompt = (
        "You are an expert assistant. Answer the following user question using ONLY the provided context. "
        "If the answer is not present in the context, say 'Not found in the provided documents.'\n"
        f"Context:\n{context}\n"
        f"Question: {question}\nAnswer: "
    )
    try:
        result = llm.complete(prompt, max_tokens=256, stop=["\n"], temperature=0.2)
        return result.text.strip()
    except Exception as e:
        return f"[Fireworks API Error] {str(e)}"
