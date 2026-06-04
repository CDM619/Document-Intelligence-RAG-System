from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi

# -------------------------
# LLM
# -------------------------

llm = OllamaLLM(model="mistral")

# -------------------------
# Embeddings
# -------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# -------------------------
# Vector DB
# -------------------------

db = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

# -------------------------
# CrossEncoder Reranker
# -------------------------

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

# -------------------------
# Retrieval Function
# -------------------------

def retrieve(query):

    # Vector Search
    vector_docs = db.max_marginal_relevance_search(
        query,
        fetch_k=25,
        k=4
    )

    # BM25 Search
    all_docs = db.get()["documents"]

    tokenized_docs = [
        doc.split() for doc in all_docs
    ]

    bm25 = BM25Okapi(tokenized_docs)

    scores = bm25.get_scores(
        query.split()
    )

    top_bm25_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:20]

    bm25_docs = [
        all_docs[i]
        for i in top_bm25_indices
    ]

    # -------------------------
    # Hybrid Merge
    # -------------------------

    merged = []

    for doc in vector_docs:
        merged.append(doc.page_content)

    merged.extend(bm25_docs)

    merged = list(set(merged))

    # -------------------------
    # CrossEncoder Reranking
    # -------------------------

    pairs = [
        (query, chunk)
        for chunk in merged
    ]

    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(merged, scores),
        key=lambda x: x[1],
        reverse=True
    )

    top_chunks = [
        chunk
        for chunk, score in ranked[:5]
    ]

    return top_chunks

# -------------------------
# Ask Function
# -------------------------

def ask(query):

    chunks = retrieve(query)

    context = "\n\n".join(chunks)

    prompt = f"""
You are a strict retrieval-based AI assistant.

Rules:
1. Answer ONLY from the context.
2. Do NOT use prior knowledge.
3. If answer is missing, say:
   "I don't know"

Context:
{context}

Question:
{query}

Answer:
"""

    return llm.invoke(prompt)

# -------------------------
# Chat Loop
# -------------------------

while True:

    q = input("\nAsk: ")

    if q.lower() == "exit":
        break

    print("\n", ask(q))