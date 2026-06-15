import os

# ==========================
# Models
# ==========================

MODEL_NAME = "llama3.2"

# Biomedical sentence transformer fine-tuned on PubMed + MS-MARCO
# Understands clinical synonyms: "myocardial infarction" ↔ "chest pain"
# Replaces: "sentence-transformers/all-MiniLM-L6-v2"
EMBED_MODEL = "pritamdeka/S-PubMedBert-MS-MARCO"

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

# Smaller chunks = tighter focus per chunk = more accurate grading
# Replaces: CHUNK_SIZE = 500
CHUNK_SIZE = 400

# Higher overlap = fewer missed sentences at chunk boundaries
# Replaces: CHUNK_OVERLAP = 50
CHUNK_OVERLAP = 80

# ==========================
# Retrieval
# ==========================

# Fetch more candidates and let the grader filter down to the best
# Replaces: TOP_K = 5
TOP_K = 8