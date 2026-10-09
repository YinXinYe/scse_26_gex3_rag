### COMPLETE THE CODE ###
import os

# Route HuggingFace model downloads through a reachable mirror
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from policy_loader import load_policy_documents

# Embedding model used to turn each policy document into a vector
_EMBEDDINGS = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Lazy-built FAISS vector store over the single knowledge-base document
_INDEX = None


def build_index():
    """Load the policy documents and index them with FAISS."""
    global _INDEX
    documents = load_policy_documents()
    _INDEX = FAISS.from_documents(documents, _EMBEDDINGS)
    return _INDEX


def get_index():
    """Return the FAISS index, building it once if necessary."""
    global _INDEX
    if _INDEX is None:
        build_index()
    return _INDEX


def retrieve_documents(query, k=3):
    """Retrieve the k most relevant policy documents for a query.

    Returns a list of (Document, score) tuples ordered by descending relevance.
    """
    return get_index().similarity_search_with_score(query, k=k)
