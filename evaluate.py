import time
from agent import run_agent


# ------------------------------------------------------------------
# Test Cases
# ------------------------------------------------------------------

test_cases = [
    {
        "query": "What are symptoms of pneumonia?",
        "expected_keyword": "cough",
        "category": "Symptoms"
    },
    {
        "query": "What is the dosage for ibuprofen?",
        "expected_keyword": "mg",
        "category": "Dosage"
    },
    {
        "query": "What are diabetes treatment guidelines?",
        "expected_keyword": "insulin",
        "category": "Treatment"
    },
    {
        "query": "What is hypertension?",
        "expected_keyword": "blood pressure",
        "category": "Definition"
    },
    {
        "query": "What medications treat malaria?",
        "expected_keyword": "artemisinin",
        "category": "Medication"
    },
    {
        "query": "What are the signs of a heart attack?",
        "expected_keyword": "chest",
        "category": "Symptoms"
    },
    {
        "query": "What is the first-line treatment for tuberculosis?",
        "expected_keyword": "rifampicin",
        "category": "Treatment"
    },
    {
        "query": "What are the side effects of metformin?",
        "expected_keyword": "gastrointestinal",
        "category": "Medication"
    },
]


# ------------------------------------------------------------------
# Runner
# ------------------------------------------------------------------

def run_evaluation():

    print("\n" + "=" * 70)
    print("  MediQuery — Evaluation Report")
    print("=" * 70)

    passed         = 0
    total          = len(test_cases)
    total_time     = 0.0
    rewrite_total  = 0

    category_stats: dict[str, dict] = {}

    results = []

    for tc in test_cases:

        start   = time.time()
        result  = run_agent(tc["query"])
        elapsed = round(time.time() - start, 2)

        answer  = result["answer"].lower()
        keyword = tc["expected_keyword"].lower()
        ok      = keyword in answer
        cat     = tc["category"]

        passed        += int(ok)
        total_time    += elapsed
        rewrite_total += result["rewrite_count"]

        # Per-category tracking

        if cat not in category_stats:
            category_stats[cat] = {"passed": 0, "total": 0}

        category_stats[cat]["total"]  += 1
        category_stats[cat]["passed"] += int(ok)

        results.append({
            "query":         tc["query"],
            "keyword":       tc["expected_keyword"],
            "ok":            ok,
            "rewrite_count": result["rewrite_count"],
            "route":         result["trace"][1] if len(result["trace"]) > 1 else "—",
            "elapsed":       elapsed,
        })

        status = "✅ PASS" if ok else "❌ FAIL"

        print(
            f"\n{status} [{cat}]"
            f"\n  Query   : {tc['query']}"
            f"\n  Expected: '{tc['expected_keyword']}'"
            f"\n  Rewrites: {result['rewrite_count']}"
            f"\n  Time    : {elapsed}s"
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    accuracy    = round(passed / total * 100)
    avg_time    = round(total_time / total, 2)
    avg_rewrite = round(rewrite_total / total, 2)

    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    print(f"  Overall Accuracy  : {passed}/{total}  ({accuracy}%)")
    print(f"  Avg Response Time : {avg_time}s per query")
    print(f"  Avg Rewrites      : {avg_rewrite} per query")

    # Per-category breakdown

    print("\n  Category Breakdown:")
    print("  " + "-" * 40)

    for cat, stats in category_stats.items():

        cat_pct = round(stats["passed"] / stats["total"] * 100)

        bar = "█" * stats["passed"] + "░" * (stats["total"] - stats["passed"])

        print(
            f"  {cat:<12} {bar}  "
            f"{stats['passed']}/{stats['total']}  ({cat_pct}%)"
        )

    # Failed cases detail

    failed = [r for r in results if not r["ok"]]

    if failed:

        print("\n  Failed Cases:")
        print("  " + "-" * 40)

        for r in failed:
            print(
                f"  • {r['query'][:55]:<55}"
                f"  keyword: '{r['keyword']}'"
            )

    else:

        print("\n  🎉 All test cases passed!")

    print("=" * 70 + "\n")

    return {
        "accuracy":    accuracy,
        "passed":      passed,
        "total":       total,
        "avg_time":    avg_time,
        "avg_rewrite": avg_rewrite,
        "results":     results,
    }


# ------------------------------------------------------------------
# Entry Point
# ------------------------------------------------------------------

if __name__ == "__main__":
    run_evaluation()