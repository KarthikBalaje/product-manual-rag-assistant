"""Compare chunking strategies on a sample PlayStation manual page."""

from __future__ import annotations

from pathlib import Path

from colorama import Fore, Style, init
from pypdf import PdfReader

from app.chunking_strategies import compare_all_strategies
from app.config import resolve_project_path

init(autoreset=True)


def find_sample_pdf(pdf_dir: Path | None = None) -> Path | None:
    directory = pdf_dir or resolve_project_path("data/pdfs")
    if not directory.exists():
        return None
    return next(iter(sorted(directory.glob("*.pdf"))), None)


def extract_sample_text(pdf_path: Path) -> str:
    try:
        reader = PdfReader(str(pdf_path))
    except Exception as exc:
        print(f"{Fore.RED}Error reading PDF: {exc}{Style.RESET_ALL}")
        return ""

    for page in reader.pages[:3]:
        try:
            text = page.extract_text()
        except Exception:
            text = ""
        if text and len(text.strip()) > 100:
            return text
    return ""


def compare_chunking(pdf_path: Path | None = None) -> int:
    print(f"{Fore.CYAN}{'=' * 60}")
    print("PLAYSTATION MANUAL CHUNKING COMPARISON")
    print(f"{'=' * 60}{Style.RESET_ALL}\n")

    sample_pdf = pdf_path or find_sample_pdf()
    if sample_pdf is None:
        print(f"{Fore.RED}No manuals found in data/pdfs/{Style.RESET_ALL}")
        return 1

    sample_text = extract_sample_text(sample_pdf)
    if not sample_text:
        print(f"{Fore.RED}No extractable text found in {sample_pdf.name}{Style.RESET_ALL}")
        return 1

    print(f"{Fore.YELLOW}File: {sample_pdf.name}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Characters: {len(sample_text)}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Words: {len(sample_text.split())}{Style.RESET_ALL}\n")

    results = compare_all_strategies(sample_text)

    for name, stats in results.items():
        print(f"{Fore.CYAN}{name}{Style.RESET_ALL}")
        print(f"  Chunks: {stats['chunks']}")
        print(f"  Avg size: {stats['avg_size']}")
        preview = stats.get("first_chunk", "")
        if preview:
            print(f"  Preview: {preview[:160]}...")
        print()

    print(f"{Fore.GREEN}Recommended default: manual chunking for product manuals.{Style.RESET_ALL}")
    return 0


def main() -> int:
    return compare_chunking()


if __name__ == "__main__":
    raise SystemExit(main())

