import os
import glob

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from config import (
    DATA_RAW,
    VECTORSTORE_PATH,
    EMBED_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP
)


def ingest_pdfs():

    pdfs = glob.glob(
        os.path.join(DATA_RAW, "*.pdf")
    )

    if not pdfs:
        print("No PDFs found in data/raw/")
        return

    all_docs = []

    for pdf in pdfs:
        print(f"Loading: {os.path.basename(pdf)}")

        loader = PyPDFLoader(pdf)

        docs = loader.load()

        # Store filename in metadata
        for doc in docs:
            doc.metadata["source"] = os.path.basename(pdf)

        all_docs.extend(docs)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    # Create chunks
    chunks = splitter.split_documents(
        all_docs
    )

    # Verify source metadata
    print(
        f"\nSources detected:"
        f"\n{set(d.metadata['source'] for d in chunks)}"
    )

    print(
        f"\nCreating embeddings using:"
        f"\n{EMBED_MODEL}"
    )

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL
    )

    print("Building FAISS vector store...")

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    os.makedirs(
        VECTORSTORE_PATH,
        exist_ok=True
    )

    vectorstore.save_local(
        VECTORSTORE_PATH
    )

    print(
        f"\n✓ Ingested {len(pdfs)} PDF(s)"
        f"\n✓ Created {len(chunks)} chunks"
        f"\n✓ Saved vector store to:"
        f"\n{VECTORSTORE_PATH}"
    )


if __name__ == "__main__":
    ingest_pdfs()