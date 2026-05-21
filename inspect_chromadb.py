import chromadb
from chromadb.config import Settings
import yaml
from pathlib import Path
from colorama import Fore, Style, init

init(autoreset=True)

# ----------------------------------------
# 🔹 LOAD CONFIG
# ----------------------------------------
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

base_path = Path(config['vectordb']['persist_directory'])

print(f"{Fore.CYAN}{'='*60}")
print(f"🔍 CHROMADB PRODUCT DATABASE INSPECTOR")
print(f"{'='*60}{Style.RESET_ALL}\n")

# ----------------------------------------
# 🔹 FIND ALL PRODUCT DATABASES
# ----------------------------------------
if not base_path.exists():
    print(f"{Fore.RED}❌ No vector database found{Style.RESET_ALL}")
    exit()

product_dirs = [p for p in base_path.iterdir() if p.is_dir()]

if not product_dirs:
    print(f"{Fore.RED}❌ No product vectorstores found{Style.RESET_ALL}")
    exit()

print(f"{Fore.YELLOW}📦 Found {len(product_dirs)} product databases{Style.RESET_ALL}\n")

# ----------------------------------------
# 🔹 LOOP THROUGH PRODUCTS
# ----------------------------------------
for product_dir in product_dirs:

    product_name = product_dir.name

    print(f"{Fore.CYAN}{'-'*60}")
    print(f"📱 Product: {product_name}")
    print(f"{'-'*60}{Style.RESET_ALL}")

    try:
        # Connect to this product DB
        client = chromadb.PersistentClient(path=str(product_dir))

        safe = "".join(ch if ch.isalnum() else "_" for ch in product_name).strip("_")
        collection_name = f"{safe}_collection"

        collection = client.get_collection(name=collection_name)

        count = collection.count()

        print(f"{Fore.YELLOW}📊 Stats:{Style.RESET_ALL}")
        print(f"   • Collection: {collection_name}")
        print(f"   • Total vectors: {count}")
        print(f"   • Path: {product_dir}\n")

        # ----------------------------------------
        # SAMPLE DOCUMENTS
        # ----------------------------------------
        results = collection.get(limit=5, include=['documents', 'metadatas'])

        print(f"{Fore.YELLOW}📄 Sample Chunks:{Style.RESET_ALL}\n")

        for i, (doc, meta) in enumerate(zip(results['documents'], results['metadatas']), 1):

            print(f"{Fore.GREEN}Chunk {i}:{Style.RESET_ALL}")
            print(f"   📄 Source: {meta.get('source', 'manual')}")
            print(f"   📄 Page: {meta.get('page', 'N/A')}")

            # 🔥 NEW METADATA
            print(f"   🧠 Section: {meta.get('section_type', 'general')}")
            print(f"   📌 Type: {meta.get('content_type', 'info')}")

            print(f"   📖 Content: {doc[:150]}...\n")

        # ----------------------------------------
        # SOURCE SUMMARY
        # ----------------------------------------
        all_data = collection.get(include=['metadatas'])

        sources = set(meta.get('source', 'Unknown') for meta in all_data['metadatas'])

        print(f"{Fore.YELLOW}📚 Sources:{Style.RESET_ALL}")

        for source in sorted(sources):
            source_count = sum(
                1 for meta in all_data['metadatas']
                if meta.get('source') == source
            )

            print(f"   • {source}: {source_count} chunks")

        print()

    except Exception as e:
        print(f"{Fore.RED}❌ Error loading {product_name}: {e}{Style.RESET_ALL}\n")
