"""Inspect the configured ChromaDB vector store."""

from __future__ import annotations

import chromadb
from chromadb.config import Settings
from colorama import Fore, Style, init

from app.config import load_config, resolve_project_path

init(autoreset=True)


def inspect_vectorstore(config_path: str = "config.yaml") -> int:
    config = load_config(config_path)
    vectorstore_path = resolve_project_path(config["vectordb"]["persist_directory"])
    collection_name = config["vectordb"]["collection_name"]

    print(f"{Fore.CYAN}{'=' * 60}")
    print("CHROMADB VECTORSTORE INSPECTOR")
    print(f"{'=' * 60}{Style.RESET_ALL}\n")

    if not vectorstore_path.exists() or not any(vectorstore_path.iterdir()):
        print(f"{Fore.RED}No vector database found at {vectorstore_path}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Run: python -m app.ingest manual{Style.RESET_ALL}")
        return 1

    try:
        client = chromadb.PersistentClient(
            path=str(vectorstore_path),
            settings=Settings(anonymized_telemetry=False),
        )
        collection = client.get_collection(name=collection_name)
    except Exception as exc:
        print(f"{Fore.RED}Unable to open collection '{collection_name}': {exc}{Style.RESET_ALL}")
        return 1

    count = collection.count()
    print(f"{Fore.YELLOW}Collection: {collection_name}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Path: {vectorstore_path}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Total vectors: {count}{Style.RESET_ALL}\n")

    results = collection.get(limit=5, include=["documents", "metadatas"])
    documents = results.get("documents") or []
    metadatas = results.get("metadatas") or []

    print(f"{Fore.CYAN}Sample Chunks{Style.RESET_ALL}")
    for index, (document, metadata) in enumerate(zip(documents, metadatas), start=1):
        print(f"\n{Fore.GREEN}Chunk {index}:{Style.RESET_ALL}")
        print(f"  Source: {metadata.get('source', 'manual')}")
        print(f"  Page: {metadata.get('page', 'N/A')}")
        print(f"  Section: {metadata.get('section_type', 'general')}")
        print(f"  Type: {metadata.get('content_type', 'informational')}")
        print(f"  Content: {document[:180]}...")

    all_data = collection.get(include=["metadatas"])
    all_metadatas = all_data.get("metadatas") or []
    source_counts: dict[str, int] = {}
    for metadata in all_metadatas:
        source = metadata.get("source", "Unknown")
        source_counts[source] = source_counts.get(source, 0) + 1

    print(f"\n{Fore.CYAN}Source Summary{Style.RESET_ALL}")
    for source, source_count in sorted(source_counts.items()):
        print(f"  {source}: {source_count} chunks")

    return 0


def main() -> int:
    return inspect_vectorstore()


if __name__ == "__main__":
    raise SystemExit(main())

