from langchain_ollama import OllamaLLM
from langchain_classic.chains import RetrievalQA

from retriever import load_retriever
from config import MODEL_NAME


def run_agent(query: str) -> str:
    """
    Run a RAG query using Ollama + FAISS.
    """

    llm = OllamaLLM(
        model=MODEL_NAME
    )

    retriever = load_retriever()

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    result = qa_chain.invoke(
        {"query": query}
    )

    answer = result["result"]
    sources = result["source_documents"]

    print("\n" + "=" * 60)
    print("QUESTION")
    print("=" * 60)
    print(query)

    print("\n" + "=" * 60)
    print("ANSWER")
    print("=" * 60)
    print(answer)

    print("\n" + "=" * 60)
    print("SOURCES")
    print("=" * 60)
    print(f"Retrieved {len(sources)} chunks")

    return answer


if __name__ == "__main__":

    run_agent(
        "What are the treatment guidelines for hypertension?"
    )