"""课程任务B：同一批Top-5向量候选的Cross-Encoder重排序。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

ROOT = Path(__file__).parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
RERANK_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
TOP_K_RETRIEVAL, TOP_K_CONTEXT = 5, 3


def read(name):
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def retrieve_top5(question, chunks, embedder, vectors):
    """仅用向量相似度选前五条，原始排名和得分均被保留。"""
    query = embedder.encode([question], normalize_embeddings=True, show_progress_bar=False)[0]
    scores = vectors @ query
    indexes = np.argsort(-scores, kind="stable")[:TOP_K_RETRIEVAL]
    if len(indexes) != 5:
        raise RuntimeError("知识库必须至少有5个片段。")
    result = []
    for rank, i in enumerate(indexes, 1):
        row = dict(chunks[int(i)])
        row.update(original_rank=rank, retrieval_score=float(scores[int(i)]))
        result.append(row)
    return result


def rerank(question, candidates, model):
    """重排器只看粗召回的同一批5条，不会访问第6条以后的知识。"""
    scores = model.predict([(question, row["text"]) for row in candidates], show_progress_bar=False)
    result = []
    for row, score in zip(candidates, scores):
        item = dict(row)
        item["rerank_score"] = float(score)
        result.append(item)
    result.sort(key=lambda row: row["rerank_score"], reverse=True)
    for rank, row in enumerate(result, 1):
        row["rerank_rank"] = rank
    return result


def show(before, after):
    print("\n向量检索 Top-5：")
    for x in before:
        print(f"  {x['original_rank']}. {x['chunk_id']} | {x['retrieval_score']:.4f}")
    print("\nCross-Encoder 重排后的同一批 Top-5：")
    for x in after:
        print(f"  {x['rerank_rank']}. {x['chunk_id']} | {x['rerank_score']:.4f} | 原第 {x['original_rank']} 名")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question-id", default="Q06")
    args = parser.parse_args()
    chunks, questions = read("policy_chunks.json"), read("questions.json")
    question = next((x for x in questions if x["question_id"] == args.question_id), None)
    if not question:
        raise SystemExit("题号不存在")
    start = perf_counter()
    print("加载 Embedding：", EMBEDDING_MODEL)
    embedder = SentenceTransformer(EMBEDDING_MODEL, cache_folder=str(ROOT / "models"))
    vectors = embedder.encode([x["text"] for x in chunks], normalize_embeddings=True, show_progress_bar=False)
    print("加载 Cross-Encoder：", RERANK_MODEL)
    reranker = CrossEncoder(RERANK_MODEL, max_length=512, cache_folder=str(ROOT / "models"))
    before = retrieve_top5(question["question"], chunks, embedder, vectors)
    after = rerank(question["question"], before, reranker)
    show(before, after)
    RESULTS.mkdir(exist_ok=True)
    record = {"question": question, "top5_before_rerank": before, "top5_after_rerank": after, "baseline_context_chunk_ids": [x["chunk_id"] for x in before[:3]], "rerank_context_chunk_ids": [x["chunk_id"] for x in after[:3]], "elapsed_seconds": round(perf_counter()-start, 3)}
    output = RESULTS / f"{args.question_id.lower()}_result.json"
    output.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n已保存：", output)


if __name__ == "__main__":
    main()
