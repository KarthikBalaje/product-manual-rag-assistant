import sys
import os
sys.path.insert(0, '.')

from app.agent import RAGAgent
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from colorama import Fore, Style, init

init(autoreset=True)

# ----------------------------------------
# 🔹 INIT
# ----------------------------------------
agent = RAGAgent()

question = input(f"{Fore.GREEN}Enter a question: {Style.RESET_ALL}")

# ----------------------------------------
# 🔹 STEP 1: EXTRACT PRODUCT
# (same logic as agent)
# ----------------------------------------
product_name = question  # simple version

vectorstore_path = f"{agent.config['vectordb']['persist_directory']}/{product_name}"
safe = "".join(ch if ch.isalnum() else "_" for ch in product_name).strip("_")
collection_name = f"{safe}_collection"

# ----------------------------------------
# 🔹 STEP 2: ENSURE INGESTION
# ----------------------------------------
if not os.path.exists(vectorstore_path):
    print(f"{Fore.YELLOW}📥 No vector DB found. Running ingestion...{Style.RESET_ALL}")

    from app.ingest import DocumentIngestion

    ingestion = DocumentIngestion(
        product_name=product_name,
        config=agent.config,
        strategy="manual"
    )
    ingestion.run()

# ----------------------------------------
# 🔹 STEP 3: LOAD VECTORSTORE
# ----------------------------------------
embeddings = HuggingFaceEmbeddings(
    model_name=agent.config["embeddings"]["model_name"]
)

db = Chroma(
    persist_directory=vectorstore_path,
    embedding_function=embeddings,
    collection_name=collection_name,
)

# ----------------------------------------
# 🔹 STEP 4: SIMILARITY SEARCH WITH SCORE
# ----------------------------------------
results = db.similarity_search_with_score(question, k=5)

print(f"\n{Fore.CYAN}🎯 Top 5 Similar Chunks:{Style.RESET_ALL}\n")

# ----------------------------------------
# 🔹 STEP 5: PRINT RESULTS
# ----------------------------------------
for i, (doc, score) in enumerate(results, 1):

    similarity = 1 - score  # convert distance → similarity

    print(f"{Fore.YELLOW}Rank {i} - Similarity: {similarity:.3f}{Style.RESET_ALL}")

    print(f"   📄 Source: {doc.metadata.get('source', 'manual')}")
    print(f"   📄 Page: {doc.metadata.get('page', 'N/A')}")

    # 🔥 NEW METADATA (from your improved chunking)
    print(f"   🧠 Section: {doc.metadata.get('section_type', 'general')}")
    print(f"   📌 Type: {doc.metadata.get('content_type', 'info')}")

    print(f"   📖 Content: {doc.page_content[:200]}...")
    print()
