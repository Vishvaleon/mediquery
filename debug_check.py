# debug_check.py
# Run after every ingestor.py to confirm keywords exist in vectorstore
# before running evaluate.py — catches missing PDFs and retrieval gaps early.

from retriever import load_retriever


# ------------------------------------------------------------------
# Keyword coverage checks
# ------------------------------------------------------------------

checks = [
    ("chloroquine malaria treatment",       "chloroquine"),
    ("chest pain heart attack symptoms",    "chest"),
    ("metformin nausea side effects",       "nausea"),
    ("insulin diabetes treatment",          "insulin"),
    ("rifampicin tuberculosis first-line",  "rifampicin"),
    ("ibuprofen dosage milligrams",         "mg"),
    ("pneumonia cough symptoms",            "cough"),
    ("blood pressure hypertension",         "blood pressure"),
]


# ------------------------------------------------------------------
# Runner
# ------------------------------------------------------------------

def run_debug_check():

    print("\n" + "=" * 60)
    print("  Vectorstore keyword coverage check")
    print("=" * 60)

    r = load_retriever()

    passed = 0
    failed = 0
    failed_items = []

    for query, keyword in checks:

        docs     = r.invoke(query)          # .get_relevant_documents() removed in new LangChain
        all_text = " ".join(
            d.page_content.lower() for d in docs
        )
        found = keyword.lower() in all_text

        source_files = list(set(
            d.metadata.get("source", "unknown")
            for d in docs
        ))

        # Shorten paths to just filenames for readability
        source_names = [
            s.split("/")[-1].split("\\")[-1]
            for s in source_files
        ]

        status = "✅" if found else "❌"

        print(
            f"\n{status} keyword : '{keyword}'"
            f"\n   query  : {query}"
            f"\n   chunks : {len(docs)}"
            f"\n   sources: {source_names}"
        )

        if found:
            passed += 1
        else:
            failed += 1
            failed_items.append((query, keyword))

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    total = len(checks)

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Passed : {passed}/{total}")
    print(f"  Failed : {failed}/{total}")

    if failed_items:
        print("\n  ❌ Missing keywords — fix before running evaluate.py:")
        print("  " + "-" * 40)
        for query, keyword in failed_items:
            print(f"  • '{keyword}'  ←  query: \"{query}\"")
        print()
        print("  Possible causes:")
        print("  1. PDF containing this topic is missing from data/raw/")
        print("  2. Keyword exists but isn't in the top retrieved chunks")
        print("     → increase TOP_K in config.py and re-run ingestor.py")
        print("  3. Embedding model mismatch")
        print("     → delete data/vectorstore/ and re-run ingestor.py")
    else:
        print("\n  🎉 All keywords found — safe to run evaluate.py")

    print("=" * 60 + "\n")

    return {
        "passed": passed,
        "failed": failed,
        "total":  total,
        "failed_items": failed_items,
    }


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    run_debug_check()