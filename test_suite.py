import sys

from colorama import Fore, Style, init

from app.agent import RAGAgent

init(autoreset=True)


def run_tests():
    print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}PLAYSTATION RAG SMOKE TEST SUITE{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}\n")

    agent = RAGAgent()

    tests = [
        {
            "name": "Console Setup Query",
            "question": "How do I set up a PlayStation console?",
            "expect": "has_evaluation",
        },
        {
            "name": "Controller Pairing Query",
            "question": "How do I connect a controller to the console?",
            "expect": "has_evaluation",
        },
        {
            "name": "Out-of-Scope Query",
            "question": "How do I build a rocket ship?",
            "expect": "fallback_or_answer",
        },
    ]

    passed = 0
    failed = 0

    for i, test in enumerate(tests, 1):
        print(f"{Fore.YELLOW}Test {i}: {test['name']}{Style.RESET_ALL}")
        print(f"Question: {test['question']}")

        try:
            result = agent.query(test["question"])
        except Exception as exc:
            print(f"{Fore.RED}FAILED - Query raised: {exc}{Style.RESET_ALL}\n")
            failed += 1
            continue

        if not result.get("answer"):
            print(f"{Fore.RED}FAILED - Empty answer returned{Style.RESET_ALL}\n")
            failed += 1
            continue

        evaluation = result.get("evaluation") or {}
        search_metrics = evaluation.get("search")
        answer_metrics = evaluation.get("answer")

        if test["expect"] == "has_evaluation":
            if not search_metrics or not answer_metrics:
                print(f"{Fore.RED}FAILED - Missing evaluation metrics{Style.RESET_ALL}\n")
                failed += 1
                continue

        if test["expect"] == "fallback_or_answer":
            fallback = "i couldn't find this in the provided playstation manual(s)." in result["answer"].lower()
            if not fallback and not result.get("sources"):
                print(f"{Fore.RED}FAILED - Neither fallback nor sources present{Style.RESET_ALL}\n")
                failed += 1
                continue

        print(f"{Fore.GREEN}PASSED{Style.RESET_ALL}")
        print(f"Answer preview: {result['answer'][:120]}...")
        print(f"Sources: {len(result.get('sources', []))}")
        print(f"Search metrics: {search_metrics}")
        print(f"Answer metrics: {answer_metrics}\n")
        passed += 1

    print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}TEST SUMMARY{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}Passed: {passed}/{len(tests)}{Style.RESET_ALL}")
    print(f"{Fore.RED}Failed: {failed}/{len(tests)}{Style.RESET_ALL}")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    run_tests()
