
from pathlib import Path
import json
import re
import sys

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mock_llm

POLICY_DIR = ROOT / "data" / "policies"
QUESTIONS_FILE = ROOT / "data" / "raw" / "policy_eval.csv"
OUTPUT_DIR = ROOT / "outputs"

CHUNK_SIZE = 40
CHUNK_OVERLAP = 10
TOP_K = 3


def split_into_chunks(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into word-based chunks with the requested overlap."""
    words = text.split()
    if not words:
        return []

    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Require chunk_size > overlap >= 0")

    step = chunk_size - overlap
    chunks = []

    for start in range(0, len(words), step):
        chunk_words = words[start:start + chunk_size]
        if not chunk_words:
            break
        chunks.append(" ".join(chunk_words))

        if start + chunk_size >= len(words):
            break

    return chunks


def load_policy_documents():
    documents = []

    for path in sorted(POLICY_DIR.glob("*.txt")):
        text = path.read_text(encoding="utf-8").strip()
        documents.append({
            "source": path.name,
            "text": text,
        })

    if len(documents) != 5:
        raise ValueError(
            f"Expected five policy documents; found {len(documents)}"
        )

    return documents


def build_chunks(documents, chunk_size, overlap):
    rows = []

    for document in documents:
        chunks = split_into_chunks(
            document["text"],
            chunk_size=chunk_size,
            overlap=overlap,
        )

        for index, chunk in enumerate(chunks):
            rows.append({
                "source": document["source"],
                "chunk_id": f'{document["source"]}__{index + 1:02d}',
                "chunk_index": index,
                "text": chunk,
                "word_count": len(chunk.split()),
            })

    return pd.DataFrame(rows)


class PolicyRetriever:
    def __init__(self, chunks):
        if chunks.empty:
            raise ValueError("Cannot build a retriever from zero chunks")

        self.chunks = chunks.reset_index(drop=True).copy()
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            token_pattern=r"(?u)\b\w+\b",
        )

        self.matrix = self.vectorizer.fit_transform(
            self.chunks["text"].tolist()
        )

    def retrieve(self, question, top_k=TOP_K):
        query_vector = self.vectorizer.transform([question])
        scores = cosine_similarity(
            query_vector, self.matrix
        ).ravel()

        # Stable tie-breaking by original chunk order.
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: (-scores[index], index),
        )[:min(top_k, len(scores))]

        results = []
        for index in ranked_indices:
            row = self.chunks.iloc[index]
            results.append({
                "source": row["source"],
                "chunk_id": row["chunk_id"],
                "text": row["text"],
                "score": float(scores[index]),
            })

        return results


def answer_question(question, retriever):
    retrieved = retriever.retrieve(question)

    context = "\n".join(
        f'[{item["source"]}] {item["text"]}'
        for item in retrieved
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are an Instamart policy assistant. "
                "Answer using only the supplied policy context. "
                "Do not invent policy details. If the answer is absent, "
                "say that the policy context does not provide the answer."
            ),
        },
        {
            "role": "user",
            "content": (
                f"CONTEXT:\n{context}\n\n"
                f"QUESTION: {question}"
            ),
        },
    ]

    response = mock_llm.chat(messages)
    answer = response.choices[0].message.content or ""

    return {
        "question": question,
        "answer": answer,
        "retrieved_sources": ";".join(
            item["source"] for item in retrieved
        ),
        "retrieved_chunk_ids": ";".join(
            item["chunk_id"] for item in retrieved
        ),
        "total_tokens": int(response.usage.total_tokens),
    }


def evaluate_questions(questions, retriever, experiment_name):
    results = []

    for question in questions:
        result = answer_question(question, retriever)
        result["experiment"] = experiment_name
        results.append(result)

    return pd.DataFrame(results)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    documents = load_policy_documents()
    questions_df = pd.read_csv(QUESTIONS_FILE, keep_default_na=False)

    if "question" not in questions_df.columns:
        raise ValueError("policy_eval.csv must contain a question column")

    questions = [
        str(question).strip()
        for question in questions_df["question"]
        if str(question).strip()
    ]

    if len(questions) != 14:
        raise ValueError(
            f"Expected 14 evaluation questions; found {len(questions)}"
        )

    # Task 1: 40-word chunks with 10-word overlap.
    overlap_chunks = build_chunks(
        documents, CHUNK_SIZE, CHUNK_OVERLAP
    )
    overlap_chunks.to_csv(
        OUTPUT_DIR / "policy_chunks_overlap.csv", index=False
    )

    print("\n--- Policy document word counts ---")
    for document in documents:
        print(
            f'{document["source"]}: '
            f'{len(document["text"].split())} words'
        )

    print("\n--- Overlapping chunk counts ---")
    print(
        overlap_chunks.groupby("source").size().to_string()
    )
    print("Total chunks:", len(overlap_chunks))

    # Task 2: TF-IDF retrieval.
    overlap_retriever = PolicyRetriever(overlap_chunks)

    retrieval_rows = []
    for question in questions:
        for rank, item in enumerate(
            overlap_retriever.retrieve(question), start=1
        ):
            retrieval_rows.append({
                "question": question,
                "rank": rank,
                **item,
            })

    pd.DataFrame(retrieval_rows).to_csv(
        OUTPUT_DIR / "policy_retrieval_results.csv",
        index=False,
    )

    # Task 3: Ask all evaluation questions using overlapping chunks.
    mock_llm.reset()
    overlap_answers = evaluate_questions(
        questions,
        overlap_retriever,
        "overlap_40_10",
    )
    overlap_answers.to_csv(
        OUTPUT_DIR / "policy_eval_overlap.csv", index=False
    )

    print("\n--- Sample policy answers ---")
    print(
        overlap_answers[
            ["question", "answer", "retrieved_sources"]
        ].head(5).to_string(index=False)
    )

    # Task 4: Compare with 40-word chunks and zero overlap.
    zero_chunks = build_chunks(documents, CHUNK_SIZE, 0)
    zero_chunks.to_csv(
        OUTPUT_DIR / "policy_chunks_no_overlap.csv", index=False
    )

    zero_retriever = PolicyRetriever(zero_chunks)

    mock_llm.reset()
    zero_answers = evaluate_questions(
        questions,
        zero_retriever,
        "no_overlap_40_0",
    )
    zero_answers.to_csv(
        OUTPUT_DIR / "policy_eval_no_overlap.csv", index=False
    )

    comparison = pd.DataFrame([
        {
            "experiment": "overlap_40_10",
            "documents": len(documents),
            "chunks": len(overlap_chunks),
            "questions": len(overlap_answers),
            "total_response_tokens": int(
                overlap_answers["total_tokens"].sum()
            ),
        },
        {
            "experiment": "no_overlap_40_0",
            "documents": len(documents),
            "chunks": len(zero_chunks),
            "questions": len(zero_answers),
            "total_response_tokens": int(
                zero_answers["total_tokens"].sum()
            ),
        },
    ])

    comparison.to_csv(
        OUTPUT_DIR / "policy_chunk_comparison.csv", index=False
    )

    print("\n--- Chunking comparison ---")
    print(comparison.to_string(index=False))

    print("\nSaved outputs:")
    for filename in [
        "policy_chunks_overlap.csv",
        "policy_retrieval_results.csv",
        "policy_eval_overlap.csv",
        "policy_chunks_no_overlap.csv",
        "policy_eval_no_overlap.csv",
        "policy_chunk_comparison.csv",
    ]:
        print(f"  outputs/{filename}")

    print("\nTasks 1–4 pipeline completed.")


if __name__ == "__main__":
    main()
