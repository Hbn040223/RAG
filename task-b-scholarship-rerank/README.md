# 任务 B：奖学金政策 RAG 重排序 Demo

## 做了什么

- 知识库：从《广东工业大学研究生奖学金评审办法》（手册印刷页 194–197）整理的、带页码来源的政策片段。
- 向量模型：`BAAI/bge-small-zh-v1.5`。
- 粗召回：固定为 Top-5。
- 重排模型：`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`。
- 严格性：Cross-Encoder 只对向量 Top-5 的同一批候选重排，不能接触额外片段。

## 真实运行结果

问题：**研究生奖学金动态管理和年度测评之间是什么关系？**

| 向量原始名次 | 片段 | 向量分数 | 重排后名次 | 重排分数 |
|---:|---|---:|---:|---:|
| 1 | P197-C01 | 0.6481 | 4 | -1.7611 |
| 2 | P196-C02 | 0.6198 | 3 | -0.9122 |
| 3 | P197-C02 | 0.6184 | 2 | -0.6322 |
| 4 | P195-C02 | 0.5914 | 1 | -0.6024 |
| 5 | P194-C03 | 0.5848 | 5 | -5.3468 |

`P195-C02` 从原第 4 名升到重排第 1 名，满足课程任务 B 的指定展示条件。向量分数和重排分数来自不同模型，只比较同一列内部的相对排名。

## 运行

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe rag_rerank.py --question-id Q06
```

首次运行会下载模型，模型缓存写入本地 `models/`，不提交到 GitHub。运行后会在 `results/q06_result.json` 保存完整的前后排序及分数。
