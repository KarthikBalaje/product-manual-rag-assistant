"""
Compare chunking strategies for Samsung Product Manuals
"""

import sys
from pathlib import Path
from pypdf import PdfReader
from colorama import Fore, Style, init

# Add project root
sys.path.insert(0, str(Path(__file__).parent))

from app.chunking_strategies import (
    AdvancedChunkingStrategies,
    compare_all_strategies
)

init(autoreset=True)


def find_sample_pdf():
    """
    Find a sample manual PDF from dynamic product folders
    """
    base_path = Path("data/pdfs")

    if not base_path.exists():
        return None, None

    # Search inside product folders
    for product_folder in base_path.iterdir():
        if product_folder.is_dir():
            pdf_files = list(product_folder.glob("*.pdf"))

            if pdf_files:
                return pdf_files[0], product_folder.name

    return None, None


def extract_sample_text(pdf_path):
    """
    Extract sample text from first few pages
    """
    try:
        reader = PdfReader(pdf_path)

        for page in reader.pages[:3]:
            text = page.extract_text()

            if text and len(text.strip()) > 100:
                return text

    except Exception as e:
        print(f"{Fore.RED}❌ Error reading PDF: {e}{Style.RESET_ALL}")

    return None


def main():

    print(f"{Fore.CYAN}{'='*60}")
    print(f"📱 SAMSUNG MANUAL CHUNKING COMPARISON")
    print(f"{'='*60}{Style.RESET_ALL}\n")

    pdf_path, product_name = find_sample_pdf()

    if not pdf_path:
        print(f"{Fore.RED}❌ No manuals found in data/pdfs/{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}💡 Run ingestion first by asking a query{Style.RESET_ALL}")
        return

    print(f"{Fore.YELLOW}📄 Product: {product_name}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}📄 File: {pdf_path.name}{Style.RESET_ALL}\n")

    sample_text = extract_sample_text(pdf_path)

    if not sample_text:
        print(f"{Fore.RED}❌ No extractable text found{Style.RESET_ALL}")
        return

    # ----------------------------------------
    # TEXT STATS
    # ----------------------------------------
    print(f"{Fore.CYAN}📊 Original Text Stats:{Style.RESET_ALL}")
    print(f"   • Characters: {len(sample_text)}")
    print(f"   • Words: {len(sample_text.split())}")
    print(f"   • Lines: {len(sample_text.splitlines())}\n")

    # ----------------------------------------
    # RUN COMPARISON
    # ----------------------------------------
    results = compare_all_strategies(sample_text)

    print(f"{Fore.CYAN}{'='*60}")
    print(f"📊 CHUNKING RESULTS")
    print(f"{'='*60}{Style.RESET_ALL}\n")

    for name, stats in results.items():
        print(f"{Fore.YELLOW}{name}:{Style.RESET_ALL}")
        print(f"   • Chunks: {stats['chunks']}")
        print(f"   • Avg size: {stats['avg_size']}")

        if stats["first_chunk"]:
            print(f"   • Preview: {stats['first_chunk'][:150]}...\n")
        else:
            print()

    # ----------------------------------------
    # MANUAL-SPECIFIC RECOMMENDATION
    # ----------------------------------------
    print(f"{Fore.CYAN}{'='*60}")
    print(f"💡 RECOMMENDATION FOR SAMSUNG MANUALS")
    print(f"{'='*60}{Style.RESET_ALL}\n")

    print(f"{Fore.GREEN}✅ BEST: manual_chunking{Style.RESET_ALL}")
    print(f"   • Preserves steps (VERY important)")
    print(f"   • Keeps troubleshooting intact")
    print(f"   • Avoids breaking instructions\n")

    print(f"{Fore.YELLOW}⚠️ Use semantic only if:{Style.RESET_ALL}")
    print(f"   • Content is simple text")
    print(f"   • Not instruction-heavy\n")

    print(f"{Fore.RED}❌ Avoid token-based for manuals:{Style.RESET_ALL}")
    print(f"   • Breaks steps randomly")
    print(f"   • Hurts retrieval quality\n")


if __name__ == "__main__":
    main()