from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from config import (
    VECTORSTORE_PATH,
    EMBED_MODEL,
    TOP_K
)


def load_retriever():

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL
    )

    vectorstore = FAISS.load_local(
        VECTORSTORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vectorstore.as_retriever(
        search_kwargs={"k": TOP_K}
    )


def retrieve(query: str):

    retriever = load_retriever()

    # LangChain 1.x API
    docs = retriever.invoke(query)

    return docs


if __name__ == "__main__":

    query = "What are symptoms of pneumonia?"

    results = retrieve(query)

    print(f"\nQuery: {query}")
    print(f"Retrieved {len(results)} chunks\n")

    for i, doc in enumerate(results, start=1):

        print("=" * 60)
        print(f"Chunk {i}")
        print("=" * 60)

        print(doc.page_content[:300])
        print()
