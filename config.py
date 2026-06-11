import os

# ==========================
# Models
# ==========================

MODEL_NAME = "llama3.2"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ==========================
# Project Paths
# ==========================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATA_RAW = os.path.join(
    BASE_DIR,
    "data",
    "raw"
)

VECTORSTORE_PATH = os.path.join(
    BASE_DIR,
    "data",
    "vectorstore"
)

# ==========================
# Text Chunking
# ==========================

CHUNK_SIZE = 500

CHUNK_OVERLAP = 50

# ==========================
# Retrieval
# ==========================

TOP_K = 5
