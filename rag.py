### COMPLETE THE CODE ###
import os

# Route HuggingFace model downloads through a reachable mirror
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from rag_tool import retrieve_documents

# Local Qwen3-0.6B model downloaded from ModelScope
MODEL_PATH = "C:/Users/Lenovo/scse26_gex3/SCSE_26_GEX3-main/models/qwen3-0.6b"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
    dtype="auto",
)
model.eval()

SYSTEM_PROMPT = (
    "You are a university IT support assistant. Answer the user's question using ONLY the "
    "retrieved policy context. If the requested fact (for example a phone number) is not "
    "present in the context, say that it is not available rather than inventing it."
)


def format_context(docs):
    """Format the retrieved documents into a compact context block."""
    blocks = []
    for i, (doc, _score) in enumerate(docs, 1):
        topic = doc.metadata.get("topic", "")
        blocks.append("[%d] topic=%s\n%s" % (i, topic, doc.page_content))
    return "\n\n".join(blocks)


def build_prompt(question, k=3):
    """Build an instruction-style prompt with the retrieved context."""
    docs = retrieve_documents(question, k=k)
    context = format_context(docs)
    return (
        SYSTEM_PROMPT
        + "\n\nRetrieved context:\n"
        + context
        + "\n\nUser: "
        + question
        + "\n\nAssistant:"
    )


def rag_answer(question, max_new_tokens=120):
    """Retrieve context and generate an answer with Qwen3-0.6B."""
    prompt = build_prompt(question)
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    decoded = tokenizer.decode(out[0], skip_special_tokens=True)
    # Return only the assistant's part of the generated text
    marker = "Assistant:"
    if marker in decoded:
        decoded = decoded.split(marker)[-1].strip()
    return decoded


if __name__ == "__main__":
    q = "What exact phone number should I call for the University IT Service Desk?"
    print("QUESTION:", q)
    print("ANSWER:", rag_answer(q))
