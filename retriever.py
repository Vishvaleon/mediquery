from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from config import (
    VECTORSTORE_PATH,
    EMBED_MODEL,
    TOP_K
)

_embeddings = None
_vectorstore = None


def load_retriever():
    global _embeddings, _vectorstore

    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL
        )

    if _vectorstore is None:
        _vectorstore = FAISS.load_local(
            VECTORSTORE_PATH,
            _embeddings,
            allow_dangerous_deserialization=True
        )

    return _vectorstore.as_retriever(
        search_kwargs={"k": TOP_K}
    )


def retrieve(query: str):
    retriever = load_retriever()
    return retriever.invoke(query)
