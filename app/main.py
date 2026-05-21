"""
Interactive CLI for the PlayStation Manual RAG assistant.
"""

import sys

from colorama import Fore, Style, init

from app.agent import RAGAgent

init(autoreset=True)


def print_banner() -> None:
    print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}PLAYSTATION MANUAL ASSISTANT{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}\n")
    print(f"{Fore.YELLOW}Ask questions based on PDFs in data/pdfs/.{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Type 'exit' or 'quit' to stop.{Style.RESET_ALL}\n")


def format_sources(sources) -> str:
    if not sources:
        return ""

    grouped = {}
    for src in sources:
        key = src.get("source", "manual")
        grouped.setdefault(key, []).append(src.get("page", "N/A"))

    lines = []
    for source, pages in grouped.items():
        pages_str = ", ".join(map(str, sorted(set(pages))))
        lines.append(f"- {source} (pages: {pages_str})")
    return "\n".join(lines)


def main() -> None:
    print_banner()

    print(f"{Fore.YELLOW}Initializing agent...{Style.RESET_ALL}")
    try:
        agent = RAGAgent()
    except Exception as exc:
        print(f"{Fore.RED}Failed to initialize agent: {exc}{Style.RESET_ALL}")
        sys.exit(1)

    print(f"{Fore.GREEN}Agent ready.{Style.RESET_ALL}\n")

    while True:
        print(f"{Fore.CYAN}{'-' * 60}{Style.RESET_ALL}")
        question = input(f"{Fore.GREEN}Question: {Style.RESET_ALL}").strip()

        if question.lower() in {"exit", "quit", "q"}:
            print(f"\n{Fore.CYAN}Goodbye.{Style.RESET_ALL}\n")
            break

        if not question:
            continue

        try:
            result = agent.query(question)
        except Exception as exc:
            print(f"{Fore.RED}Error: {exc}{Style.RESET_ALL}\n")
            continue

        print(f"\n{Fore.CYAN}Answer:{Style.RESET_ALL}")
        print(f"{Fore.WHITE}{result['answer']}{Style.RESET_ALL}\n")

        if result.get("sources"):
            print(f"{Fore.CYAN}Sources:{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}{format_sources(result['sources'])}{Style.RESET_ALL}\n")


if __name__ == "__main__":
    main()

