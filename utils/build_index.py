import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent.parent
RAG_DIR = BASE_DIR / "rag_model_data"
KNOWLEDGE_PATH = RAG_DIR / "influenza_rag_knowledge.json"
INDEX_PATH = RAG_DIR / "influenza.index"

with open(KNOWLEDGE_PATH, "r", encoding="utf-8") as f:
    docs = json.load(f)

texts = [d.get("content", "") for d in docs]

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(texts, show_progress_bar=True)

dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(np.array(embeddings))

faiss.write_index(index, str(INDEX_PATH))
print(f"✅ FAISS index built at {INDEX_PATH}")
