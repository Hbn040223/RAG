"""用相同 Qwen 模型比较无重排和有重排的最终回答。"""
from pathlib import Path
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import CrossEncoder, SentenceTransformer
from rag_rerank import retrieve_top5, rerank, EMBEDDING_MODEL, RERANK_MODEL

ROOT, DATA = Path(__file__).parent, Path(__file__).parent / "data"
question = json.loads((DATA / "questions.json").read_text(encoding="utf-8"))[0]["question"]
chunks = json.loads((DATA / "policy_chunks.json").read_text(encoding="utf-8"))
embedder = SentenceTransformer(EMBEDDING_MODEL, cache_folder=str(ROOT / "models"))
vectors = embedder.encode([x["text"] for x in chunks], normalize_embeddings=True, show_progress_bar=False)
reranker = CrossEncoder(RERANK_MODEL, max_length=512, cache_folder=str(ROOT / "models"))
before = retrieve_top5(question, chunks, embedder, vectors)
after = rerank(question, before, reranker)

def prompt(contexts):
    evidence = "\n\n".join(f"【{x['chunk_id']}】{x['text']}" for x in contexts)
    return "只能根据以下政策证据回答；证据不足则说无法确定。\n" + evidence + "\n问题：" + question

+model_path = ROOT / "models" / "qwen2.5-0.5b-instruct"
+if not model_path.exists():
+    raise SystemExit("请先运行 download_qwen_from_modelscope.py 下载本地 Qwen 模型。")
+tokenizer = AutoTokenizer.from_pretrained(model_path)
+model = AutoModelForCausalLM.from_pretrained(model_path).eval()
+
+def answer(contexts):
+    chat = tokenizer.apply_chat_template([{"role":"user", "content":prompt(contexts)}], tokenize=False, add_generation_prompt=True)
+    inputs = tokenizer(chat, return_tensors="pt")
+    with torch.no_grad():
+        output = model.generate(**inputs, do_sample=False, max_new_tokens=160, pad_token_id=tokenizer.eos_token_id)
+    return tokenizer.decode(output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
+
+print("无 Rerank（向量 Top-3）:\n", answer(before[:3]))
+print("\n有 Rerank（重排 Top-3）:\n", answer(after[:3]))
