"""
Advanced chunking strategies for user-manual RAG.
Optimized for manuals with steps, troubleshooting, and structured content.
"""

from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    TokenTextSplitter
)
from langchain_openai import ChatOpenAI
from typing import List, Optional
from langchain.schema import Document
import os
from dotenv import load_dotenv

load_dotenv()


# ============================================
# 🔹 HELPER FUNCTIONS (NEW)
# ============================================

def detect_section_type(text: str):
    text = text.lower()

    if "troubleshoot" in text:
        return "troubleshooting"
    elif "reset" in text:
        return "reset"
    elif "install" in text or "setup" in text:
        return "setup"
    elif "warning" in text or "caution" in text:
        return "warning"
    return "general"


def detect_content_type(text: str):
    if "step" in text.lower() or "1." in text:
        return "procedural"
    return "informational"


# ============================================
# 🔹 MAIN STRATEGIES
# ============================================

class AdvancedChunkingStrategies:

    # ----------------------------------------
    # 🔹 TOKEN BASED
    # ----------------------------------------
    @staticmethod
    def token_based_with_overlap(
        chunk_size: int = 512,
        chunk_overlap: int = 128,
        encoding_name: str = "cl100k_base"
    ):
        return TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            encoding_name=encoding_name
        )

    # ----------------------------------------
    # 🔹 SEMANTIC (GENERAL)
    # ----------------------------------------
    @staticmethod
    def semantic_chunking(
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        separators = [
            "\n\n\n",
            "\n\n",
            "\n",
            ". ",
            "! ",
            "? ",
            "; ",
            ", ",
            " ",
            ""
        ]

        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            keep_separator=True
        )

    # ----------------------------------------
    # 🔹 🧠 MANUAL-AWARE (DEFAULT - IMPORTANT)
    # ----------------------------------------
    @staticmethod
    def manual_chunking(
        chunk_size: int = 1200,
        chunk_overlap: int = 200
    ):
        """
        Optimized for user manuals:
        - Keeps steps intact
        - Preserves troubleshooting sections
        - Avoids breaking instructions
        """

        separators = [
            "\n\n\n",                 # Sections
            "\n\n",                   # Paragraphs
            "\n• ",                   # Bullet points
            "\n- ",                   # Lists
            "\n1. ", "\n2. ", "\n3. ",
            "\nStep ",                # Steps
            "\nWARNING",              # Warnings
            "\nCAUTION",
            "\nNOTE",
            "\n",
            ". ",
            " "
        ]

        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            keep_separator=True
        )

    # ----------------------------------------
    # 🔹 RECURSIVE
    # ----------------------------------------
    @staticmethod
    def recursive_chunking(
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        separators = ["\n\n", "\n", ". ", " ", ""]

        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators
        )

    # ----------------------------------------
    # 🔹 AGENTIC (LLM BASED)
    # ----------------------------------------
    @staticmethod
    def agentic_chunking(
        llm: Optional[ChatOpenAI] = None,
        max_chunk_size: int = 1500,
        overlap: int = 200
    ):
        if llm is None:
            llm = ChatOpenAI(
                model="gpt-4.1-mini",
                temperature=0
            )

        return AgenticChunker(llm, max_chunk_size, overlap)


# ============================================
# 🔹 AGENTIC CHUNKER (UPDATED PROMPT)
# ============================================

class AgenticChunker:

    def __init__(self, llm: ChatOpenAI, max_chunk_size=1500, overlap=200):
        self.llm = llm
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap

    def split_text(self, text: str) -> List[str]:

        if len(text) <= self.max_chunk_size:
            return [text]

        prompt = f"""
You are splitting a product user manual into chunks.

STRICT RULES:
1. DO NOT break step-by-step instructions
2. Keep troubleshooting steps intact
3. Preserve setup/reset procedures fully
4. Each chunk must represent a complete task

Target size: {self.max_chunk_size}

Text:
{text[:5000]}

Return ONLY split indices:
"""

        try:
            response = self.llm.invoke(prompt)
            indices = [int(x.strip()) for x in response.content.split(",")]

            chunks = []
            start = 0

            for idx in indices:
                chunk_start = max(0, start - self.overlap)
                chunks.append(text[chunk_start:idx].strip())
                start = idx

            if start < len(text):
                chunk_start = max(0, start - self.overlap)
                chunks.append(text[chunk_start:].strip())

            return chunks

        except Exception:
            fallback = AdvancedChunkingStrategies.semantic_chunking(
                self.max_chunk_size, self.overlap
            )
            return fallback.split_text(text)

    def split_documents(self, documents: List[Document]) -> List[Document]:

        chunked_docs = []

        for doc in documents:
            chunks = self.split_text(doc.page_content)

            for i, chunk in enumerate(chunks):
                chunked_docs.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            **doc.metadata,
                            "chunk_index": i,
                            "total_chunks": len(chunks),

                            # 🔥 NEW METADATA
                            "section_type": detect_section_type(chunk),
                            "content_type": detect_content_type(chunk)
                        }
                    )
                )

        return chunked_docs


def compare_all_strategies(sample_text: str):
    """
    Utility used by compare_chunking.py.
    Returns basic stats for each chunking strategy on the given text.
    """
    strategies = {
        "token_based": AdvancedChunkingStrategies.token_based_with_overlap(512, 128),
        "semantic": AdvancedChunkingStrategies.semantic_chunking(1000, 200),
        "manual": AdvancedChunkingStrategies.manual_chunking(1200, 200),
        "recursive": AdvancedChunkingStrategies.recursive_chunking(1000, 200),
    }

    results = {}
    for name, splitter in strategies.items():
        try:
            chunks = splitter.split_text(sample_text)
        except Exception:
            # Some splitters may only support split_documents; fall back to semantic behavior
            chunks = AdvancedChunkingStrategies.semantic_chunking(1000, 200).split_text(sample_text)

        sizes = [len(c) for c in chunks] if chunks else [0]
        results[name] = {
            "chunks": len(chunks),
            "avg_size": int(sum(sizes) / max(1, len(sizes))),
            "first_chunk": chunks[0] if chunks else "",
        }

    return results
