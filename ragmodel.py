import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
import torch
import os

# ChromaDB Collection Path
COLLECTION_DIR = os.path.join(os.path.dirname(__file__), "collection_el")

class MultilingualSentenceTransformer(EmbeddingFunction):
    """Create a ChromaDB embedding function based on the chosen model."""
    def __init__(self, model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        self.model = model
        self.embedder = self.initialize_model(model)

    def initialize_model(self, model):
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer(model, device="cuda" if torch.cuda.is_available() else "cpu")
        return embedder

    def __call__(self, sentences: Documents) -> Embeddings:
        return self.embedder.encode(sentences, convert_to_numpy=True).tolist()

def load_collection(collection_path):
    """
    Load a ChromaDB collection from the specified path.
    """
    if not os.path.exists(collection_path):
        raise FileNotFoundError(f"Collection directory not found: {collection_path}")
    client = chromadb.PersistentClient(path=collection_path)
    return client

def query_greek_with_rag(query, collection_name="collection_el", n_results=5, max_length=500):
    """
    Query the Greek dataset using RAG and a ChromaDB collection.
    """
    # Load the collection
    client = load_collection(COLLECTION_DIR)
    collection = client.get_collection(collection_name)

    # Retrieve passages
    results = collection.query(query_texts=[query], n_results=n_results)
    documents = results.get("documents", [])
    metadatas = results.get("metadatas", [])

    # Truncate documents to max_length
    truncated_docs = [doc[:max_length] for doc in documents[0]] if documents else []

    # Prepare the prompt for the LLM
    passages = "\n\n".join(truncated_docs)
    sources = {metadata.get("source", "Unknown") for metadata in metadatas[0]}
    prompt = f"Passages:\n\n{passages}\n\nQuery: {query}"

    # Call the language model
    response = query_meltemi(prompt)

    # Add sources to the response
    sources_text = "\n\nΠηγές:\n" + "\n".join(sources) if sources else "\n\nΠηγές: Καμία."
    return response + sources_text

# Meltemi setup
from openai import OpenAI
MELTEMI_API_KEY = "yousk-RYF0g_hDDIa2TLiHFboZ1Q" 
MELTEMI_BASE_URL = "http://ec2-3-19-37-251.us-east-2.compute.amazonaws.com:4000/"
MELTEMI_CLIENT = OpenAI(api_key=MELTEMI_API_KEY, base_url=MELTEMI_BASE_URL)

def query_meltemi(prompt):
    """Query the Meltemi model with the formatted prompt."""
    response = MELTEMI_CLIENT.chat.completions.create(
        model="meltemi",
        messages=[
            {"role": "system", "content": "Respond with detailed information based on the provided passages."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=3000,
    )
    return response.choices[0].message.content
