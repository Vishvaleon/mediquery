# diagnose.py
from retriever import load_retriever
from config import VECTORSTORE_PATH, EMBED_MODEL
import os

print("\n" + "=" * 60)
print("  MediQuery — Vectorstore Diagnostic")
print("=" * 60)

# Check 1: vectorstore exists
print("\n[1] Vectorstore path:", VECTORSTORE_PATH)
index_file = os.path.join(VECTORSTORE_PATH, "index.faiss")
print("    index.faiss exists:", os.path.exists(index_file))
print("    Embedding model:   ", EMBED_MODEL)

# Check 2: load retriever
try:
    r = load_retriever()
    print("\n[2] Retriever loaded OK")
except Exception as e:
    print(f"\n[2] FAILED to load retriever: {e}")
    exit()

# Check 3: keyword presence in raw chunks
failing_queries = [
    ("ibuprofen dosage milligrams adults",       "mg",               "Ibuprofen"),
    ("rifampicin tuberculosis first-line RHZE",  "rifampicin",       "TB"),
    ("metformin gastrointestinal side effects",  "gastrointestinal", "Metformin"),
]

print("\n[3] Keyword scan — top 8 chunks per query")
print("-" * 60)

for query, keyword, label in failing_queries:

    docs     = r.invoke(query)
    all_text = " ".join(d.page_content.lower() for d in docs)
    found    = keyword.lower() in all_text
    sources  = list(set(
        d.metadata.get("source", "unknown") for d in docs
    ))

    print(f"\n{'✅' if found else '❌'} [{label}] keyword: '{keyword}'")
    print(f"   Sources retrieved: {sources}")

    if not found:
        # Show what keywords ARE in the chunks instead
        words   = set(all_text.split())
        medical = [w for w in words if len(w) > 6 and w.isalpha()][:15]
        print(f"   Keywords found in chunks instead: {medical}")
    else:
        # Show the exact chunk that contains the keyword
        for d in docs:
            if keyword in d.page_content.lower():
                idx     = d.page_content.lower().find(keyword)
                snippet = d.page_content[max(0, idx - 80): idx + 120]
                print(f"   Found in: {d.metadata.get('source', '?')} "
                      f"p.{d.metadata.get('page', '?')}")
                print(f"   Context: ...{snippet}...")
                break

# Check 4: list all PDFs in data/raw
print("\n[4] PDFs in data/raw/")
print("-" * 60)

raw_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data", "raw"
)

pdfs = [f for f in os.listdir(raw_dir) if f.endswith(".pdf")]

if pdfs:
    for f in pdfs:
        size = os.path.getsize(os.path.join(raw_dir, f))
        print(f"   {f}  ({round(size / 1024)}KB)")
else:
    print("   NO PDFs found in data/raw/ !")

print("\n" + "=" * 60)