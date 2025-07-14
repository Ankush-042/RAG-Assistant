import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# You can change this model to another small, efficient open LLM if you prefer
DEFAULT_MODEL = "facebook/bart-base"

_llm_pipeline = None

def get_llm_pipeline(model_name: str = DEFAULT_MODEL, device: str = None):
    """
    Loads or returns a local HuggingFace LLM pipeline for answer synthesis.
    """
    global _llm_pipeline
    if _llm_pipeline is not None:
        return _llm_pipeline
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16 if device=="cuda" else torch.float32)
    _llm_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if device=="cuda" else -1, max_new_tokens=256)
    return _llm_pipeline

def synthesize_answer_with_llm(question: str, context: str, model_name: str = DEFAULT_MODEL) -> str:
    """
    Use the local LLM to synthesize a concise answer from retrieved context.
    Ensures the context never exceeds the model's max input length (2048 tokens for TinyLlama).
    """
    llm = get_llm_pipeline(model_name)
    tokenizer = llm.tokenizer
    max_tokens = 2048

    # Truncate context to fit within max_tokens (including prompt and question)
    prompt_base = (
        "You are an expert assistant. Answer the following user question using ONLY the provided context. "
        "If the answer is not present in the context, say 'Not found in the provided documents.'\n"
        f"Context:\n"
    )
    prompt_question = f"\nQuestion: {question}\nAnswer: "
    # Split context into lines (chunks or sentences)
    context_lines = context.split('\n')
    truncated_context = []
    total_tokens = len(tokenizer.encode(prompt_base + prompt_question))
    for line in context_lines:
        line_tokens = len(tokenizer.encode(line + '\n'))
        if total_tokens + line_tokens < max_tokens:
            truncated_context.append(line)
            total_tokens += line_tokens
        else:
            break
    safe_context = '\n'.join(truncated_context)
    prompt = prompt_base + safe_context + prompt_question
    result = llm(prompt, do_sample=False, temperature=0.2, top_k=20, num_return_sequences=1)
    return result[0]["generated_text"].split("Answer:")[-1].strip().split("\n")[0] if result else "Not found in the provided documents."
