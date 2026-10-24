# Name That LLM: сабмит = среднее 5 LoRA Qwen3-8B (24 слоя, по фолдам) + TF-IDF + LogReg + центрирование по темам теста.
# Kaggle: GPU T4 x2, интернет выключен. Входы: данные соревнования, модель qwen-lm/qwen-3/Transformers/8b-base,
# выводы ноутбуков ivanmitochondrion/name-that-llm-lora-folds-a и -b (папки lora_f0..lora_f4).
import glob, json, os
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import AgglomerativeClustering
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
import peft.import_utils, peft.tuners.lora.torchao
peft.import_utils.is_torchao_available = peft.tuners.lora.torchao.is_torchao_available = lambda: False
from peft import PeftModel

W_LORA, W_TFIDF = 1.0, 1.0               # веса смеси log-вероятностей; holdout (фолд 0): LoRA v2 0.450, TF-IDF 0.344, смесь 0.469
CENTER_THR, CENTER_W = 0.9, 2.0          # порог косинусного расстояния для тем и сила центрирования

def find(pattern):
    return sorted(glob.glob(f"/kaggle/input/**/{pattern}", recursive=True))[0]

train = pd.read_csv(find("train.csv"))
test = pd.read_csv(find("test.csv"))
test["text"] = test["text"].fillna("").astype(str)
LORAS = sorted(os.path.dirname(p) for p in glob.glob("/kaggle/input/**/lora_f*/adapter_config.json", recursive=True))
meta = json.load(open(f"{LORAS[0]}/meta.json"))
LABELS = meta["labels"]
BASE = os.path.dirname(next(p for p in glob.glob("/kaggle/input/**/config.json", recursive=True)
                            if "8b" in p.lower() and "qwen" in p.lower() and "lora" not in p.lower()))
print("train", train.shape, "test", test.shape, "| loras", len(LORAS), "| base", BASE, flush=True)

# 1. TF-IDF + логистическая регрессия
vecs = [TfidfVectorizer(analyzer="char", ngram_range=(1, 5), sublinear_tf=True, min_df=2, lowercase=False, max_features=300000),
        TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True, min_df=2, lowercase=False,
                        token_pattern=r"(?u)\b\w+\b|[^\w\s]")]
X_tr = sp.hstack([v.fit_transform(train.text) for v in vecs]).tocsr()
X_te = sp.hstack([v.transform(test.text) for v in vecs]).tocsr()
clf = LogisticRegression(C=20, max_iter=3000).fit(X_tr, train.model)
assert list(clf.classes_) == LABELS
S_tfidf = clf.predict_log_proba(X_te)

# 2. LoRA Qwen3-8B
tok = AutoTokenizer.from_pretrained(BASE)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
cfg = AutoConfig.from_pretrained(BASE, num_hidden_layers=meta["n_layers"], num_labels=len(LABELS), pad_token_id=tok.pad_token_id)
model = AutoModelForSequenceClassification.from_pretrained(BASE, config=cfg, torch_dtype=torch.float16, device_map="auto",
                                                           max_memory={0: "13GiB", 1: "13GiB"})
model = PeftModel.from_pretrained(model, LORAS[0], adapter_name="f0")
for j, path in enumerate(LORAS[1:], 1):
    model.load_adapter(path, adapter_name=f"f{j}")
model.eval()
dev = model.get_input_embeddings().weight.device
S_lora = np.zeros((len(test), len(LABELS)), np.float32)
order = np.argsort(test.text.str.len().values)        # батчи из текстов близкой длины — быстрее
for j in range(len(LORAS)):
    model.set_adapter(f"f{j}")
    with torch.no_grad():
        for i in range(0, len(order), 16):
            idx = order[i:i + 16]
            enc = tok(test.text.values[idx].tolist(), padding=True, truncation=True, max_length=meta["max_len"], return_tensors="pt")
            with torch.autocast("cuda", dtype=torch.float16):
                S_lora[idx] += torch.log_softmax(model(**{k: v.to(dev) for k, v in enc.items()}).logits.float(), -1).cpu().numpy() / len(LORAS)

# 3. смесь + центрирование внутри тем (кластеры по порогу; одиночки и группы >18 не трогаем)
S = W_LORA * S_lora + W_TFIDF * S_tfidf
if len(test) > 1:
    T = TfidfVectorizer(sublinear_tf=True, stop_words="english").fit_transform(test.text).toarray()
    grp = AgglomerativeClustering(n_clusters=None, distance_threshold=CENTER_THR, metric="cosine", linkage="average").fit_predict(T)
    for k in np.unique(grp):
        m = grp == k
        if 2 <= m.sum() <= 18:
            S[m] = S[m] + CENTER_W * (S[m] - S[m].mean(0))
    print("темы:", pd.Series(grp).value_counts().value_counts().sort_index().to_dict(), flush=True)

sub = pd.DataFrame({"id": test.id, "model": np.array(LABELS)[S.argmax(1)]})
assert sub.id.is_unique and sub.model.isin(LABELS).all() and len(sub) == len(test)
sub.to_parquet("/kaggle/working/submission.parquet", index=False)
print(sub.model.value_counts())
