"""下载本地 Qwen2.5-0.5B-Instruct，并校验权重 SHA256。"""
import hashlib, shutil, urllib.request
from pathlib import Path
ROOT = Path(__file__).parent
TARGET = ROOT / "models" / "qwen2.5-0.5b-instruct"
REV = "13448952dbdab7a1627d0680ecd207535d889a23"
FILES = {"config.json":None, "generation_config.json":None, "tokenizer.json":None, "tokenizer_config.json":None, "vocab.json":None, "merges.txt":None, "model.safetensors":"fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe"}
def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda:f.read(1024*1024), b""): h.update(block)
    return h.hexdigest()
TARGET.mkdir(parents=True, exist_ok=True)
for name, expected in FILES.items():
    out, part = TARGET/name, TARGET/(name+".part")
    if out.exists() and (not expected or digest(out) == expected): continue
    url = f"https://modelscope.cn/models/Qwen/Qwen2.5-0.5B-Instruct/resolve/{REV}/{name}"
    with urllib.request.urlopen(url, timeout=120) as response, open(part, "wb") as f: shutil.copyfileobj(response, f)
    if expected and digest(part) != expected: raise RuntimeError("SHA256 校验失败："+name)
    part.replace(out)
    print("完成：", name)
