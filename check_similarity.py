"""Run a similarity search against the configured vector store."""

from __future__ import annotations

from colorama import Fore, Style, init
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from app.config import load_config, resolve_project_path

init(autoreset=True)


def check_similarity(question: str, config_path: str = "config.yaml") -> int:
    question = (question or "").strip()
    if not question:
        print(f"{Fore.RED}Question must not be empty.{Style.RESET_ALL}")
        return 1

    config = load_config(config_path)
    vectorstore_path = resolve_project_path(config["vectordb"]["persist_directory"])
    if not vectorstore_path.exists() or not any(vectorstore_path.iterdir()):
        print(f"{Fore.RED}No vector database found at {vectorstore_path}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Run: python -m app.ingest manual{Style.RESET_ALL}")
        return 1

    embeddings = HuggingFaceEmbeddings(model_name=config["embeddings"]["model_name"])
    db = Chroma(
        persist_directory=str(vectorstore_path),
        embedding_function=embeddings,
        collection_name=config["vectordb"]["collection_name"],
    )

    results = db.similarity_search_with_score(question, k=5)
    if not results:
        print(f"{Fore.YELLOW}No similar chunks found.{Style.RESET_ALL}")
        return 0

    print(f"\n{Fore.CYAN}Top Similar Chunks{Style.RESET_ALL}\n")
    for index, (doc, score) in enumerate(results, start=1):
        similarity = 1 / (1 + max(float(score), 0.0))
        print(f"{Fore.YELLOW}Rank {index} - Similarity: {similarity:.3f}{Style.RESET_ALL}")
        print(f"  Source: {doc.metadata.get('source', 'manual')}")
        print(f"  Page: {doc.metadata.get('page', 'N/A')}")
        print(f"  Section: {doc.metadata.get('section_type', 'general')}")
        print(f"  Type: {doc.metadata.get('content_type', 'informational')}")
        print(f"  Content: {doc.page_content[:220]}...\n")

    return 0


def main() -> int:
    question = input(f"{Fore.GREEN}Enter a question: {Style.RESET_ALL}")
    return check_similarity(question)


if __name__ == "__main__":
    raise SystemExit(main())

