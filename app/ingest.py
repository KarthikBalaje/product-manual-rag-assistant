"""
Local PDF ingestion pipeline for the PlayStation manual RAG assistant.

Reads one or more PDFs from data/pdfs/, chunks them, embeds them (local embeddings),
and persists a ChromaDB collection to vectorstore/.
"""

import sys
from pathlib import Path

# Allow running as `python app/ingest.py ...` while still supporting `from app...` imports.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml
from typing import List

from dotenv import load_dotenv
from pypdf import PdfReader

from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from colorama import Fore, Style, init

from app.chunking_strategies import AdvancedChunkingStrategies

init(autoreset=True)
load_dotenv()


class DocumentIngestion:
    def __init__(self, config_path: str = "config.yaml", strategy: str = "manual"):
        print(f"{Fore.CYAN}PlayStation Manual Ingestion{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Strategy: {strategy.upper()}{Style.RESET_ALL}\n")

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.strategy_name = strategy
        self.text_splitter = self._get_chunking_strategy(strategy)

        self.pdf_dir = Path("data/pdfs")
        self.vectorstore_dir = Path(self.config["vectordb"]["persist_directory"])
        self.collection_name = self.config["vectordb"].get("collection_name", "playstation_manual")

        print(f"{Fore.YELLOW}Loading embeddings...{Style.RESET_ALL}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.config["embeddings"]["model_name"]
        )
        print(f"{Fore.GREEN}Embeddings ready{Style.RESET_ALL}\n")

    def _get_chunking_strategy(self, strategy: str):
        chunk_size = self.config["chunking"]["chunk_size"]
        chunk_overlap = self.config["chunking"]["chunk_overlap"]

        strategies = {
            "token_based": AdvancedChunkingStrategies.token_based_with_overlap(512, 128),
            "semantic": AdvancedChunkingStrategies.semantic_chunking(chunk_size, chunk_overlap),
            "manual": AdvancedChunkingStrategies.manual_chunking(1200, 200),
            "agentic": AdvancedChunkingStrategies.agentic_chunking(1500, 200),
            "recursive": AdvancedChunkingStrategies.recursive_chunking(chunk_size, chunk_overlap),
        }

        return strategies.get(strategy, strategies["manual"])

    def _find_pdfs(self) -> List[Path]:
        if not self.pdf_dir.exists():
            return []
        return sorted([p for p in self.pdf_dir.glob("*.pdf") if p.is_file()])

    def load_pdfs(self) -> List[Document]:
        pdfs = self._find_pdfs()
        if not pdfs:
            print(f"{Fore.RED}No PDFs found in {self.pdf_dir}{Style.RESET_ALL}")
            return []

        documents: List[Document] = []
        for pdf_path in pdfs:
            print(f"{Fore.CYAN}Reading: {pdf_path.name}{Style.RESET_ALL}")
            reader = PdfReader(str(pdf_path))
            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text()
                if text and text.strip():
                    documents.append(
                        Document(
                            page_content=text,
                            metadata={
                                "source": pdf_path.name,
                                "page": page_num,
                            },
                        )
                    )

        print(f"{Fore.GREEN}Loaded {len(documents)} pages total{Style.RESET_ALL}\n")
        return documents

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        print(f"{Fore.CYAN}Chunking...{Style.RESET_ALL}")
        chunks = self.text_splitter.split_documents(documents)
        print(f"{Fore.GREEN}{len(chunks)} chunks created{Style.RESET_ALL}\n")
        return chunks

    def create_vectorstore(self, chunks: List[Document]) -> None:
        print(f"{Fore.CYAN}Creating / updating vector DB...{Style.RESET_ALL}")

        Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            collection_name=self.collection_name,
            persist_directory=str(self.vectorstore_dir),
        )

        print(f"{Fore.GREEN}Stored at: {self.vectorstore_dir}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}Collection: {self.collection_name}{Style.RESET_ALL}\n")

    def run(self) -> None:
        print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}PLAYSTATION MANUAL INGESTION{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}\n")

        docs = self.load_pdfs()
        if not docs:
            return

        chunks = self.chunk_documents(docs)
        self.create_vectorstore(chunks)
        print(f"{Fore.GREEN}INGESTION COMPLETE{Style.RESET_ALL}\n")


if __name__ == "__main__":
    strategy = sys.argv[1] if len(sys.argv) > 1 else "manual"
    ingestion = DocumentIngestion(strategy=strategy)
    ingestion.run()
