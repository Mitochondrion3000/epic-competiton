# Name That LLM: дообучение Qwen3-8B-Base (первые 24 слоя) с LoRA как классификатора на 18 классов.
# Kaggle: GPU T4 x2. Вход: данные соревнования + модель qwen-lm/qwen-3/Transformers/8b-base.
# 1) обучаем на 4/5 промптов, считаем balanced accuracy на отложенной 1/5 (GroupKFold по prompt_id, фолд 0);
# 2) обучаем на всех данных и сохраняем адаптер для ноутбука-сабмита.
import glob, json, math, os, time
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model
import peft.import_utils, peft.tuners.lora.torchao
# в образе Kaggle torchao 0.10 несовместима с peft и роняет get_peft_model; torchao нам не нужна
peft.import_utils.is_torchao_available = peft.tuners.lora.torchao.is_torchao_available = lambda: False

T0 = time.time()
N_LAYERS, MAX_LEN, EPOCHS, LR, BS, ACCUM = 24, 400, 2, 1e-4, 4, 4
OUT = "/kaggle/working"

def find(pattern):
    return sorted(glob.glob(f"/kaggle/input/**/{pattern}", recursive=True))[0]

train = pd.read_csv(find("train.csv"))
LABELS = sorted(train.model.unique())
y_all = train.model.map({l: i for i, l in enumerate(LABELS)}).values
BASE = os.path.dirname(next(p for p in glob.glob("/kaggle/input/**/config.json", recursive=True)
                            if "8b" in p.lower() and "qwen" in p.lower()))
print("base:", BASE, flush=True)

tok = AutoTokenizer.from_pretrained(BASE)
tok.padding_side = "left" if tok.padding_side == "left" else "right"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
lens = [len(t) for t in tok(train.text.tolist())["input_ids"]]
print("tokens: median", int(np.median(lens)), "max", max(lens), flush=True)


def build():
    cfg = AutoConfig.from_pretrained(BASE, num_hidden_layers=N_LAYERS, num_labels=len(LABELS), pad_token_id=tok.pad_token_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE, config=cfg, torch_dtype=torch.float16, device_map="auto",
        max_memory={0: "11GiB", 1: "13GiB"})
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    lcfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="SEQ_CLS",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lcfg)
    for p in model.parameters():          # обучаемые веса в fp32, база в fp16 (T4 не умеет bf16)
        if p.requires_grad:
            p.data = p.data.float()
    model.print_trainable_parameters()
    return model


def batches(texts, labels=None, bs=BS, shuffle=False, seed=0):
    order = np.random.default_rng(seed).permutation(len(texts)) if shuffle else np.arange(len(texts))
    for i in range(0, len(order), bs):
        idx = order[i:i + bs]
        enc = tok([texts[j] for j in idx], padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
        yield idx, enc, (torch.tensor(labels[idx]) if labels is not None else None)


def fit(texts, labels):
    model = build()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.01)
    steps = EPOCHS * math.ceil(len(texts) / BS / ACCUM)
    sched = get_cosine_schedule_with_warmup(opt, int(0.1 * steps), steps)
    scaler = torch.amp.GradScaler("cuda")
    dev = model.get_input_embeddings().weight.device
    model.train()
    for ep in range(EPOCHS):
        tot, n = 0.0, 0
        for b, (idx, enc, yb) in enumerate(batches(texts, labels, shuffle=True, seed=ep)):
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(**{k: v.to(dev) for k, v in enc.items()}).logits.float()
            loss = torch.nn.functional.cross_entropy(logits, yb.to(logits.device)) / ACCUM
            scaler.scale(loss).backward()
            if (b + 1) % ACCUM == 0:
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(); sched.step()
            tot += loss.item() * ACCUM; n += 1
            if b % 100 == 0:
                print(f"ep {ep} batch {b} loss {tot / n:.4f} t={time.time() - T0:.0f}s", flush=True)
    return model


@torch.no_grad()
def predict(model, texts):
    model.eval()
    dev = model.get_input_embeddings().weight.device
    out = np.zeros((len(texts), len(LABELS)), np.float32)
    for idx, enc, _ in batches(texts, bs=16):
        with torch.autocast("cuda", dtype=torch.float16):
            out[idx] = torch.log_softmax(model(**{k: v.to(dev) for k, v in enc.items()}).logits.float(), -1).cpu().numpy()
    return out


texts = train.text.tolist()
tr, va = next(GroupKFold(5).split(train, y_all, train.prompt_id))
model = fit([texts[i] for i in tr], y_all[tr])
P = predict(model, [texts[i] for i in va])
ba = balanced_accuracy_score(y_all[va], P.argmax(1))
print(f"HOLDOUT balanced accuracy: {ba:.4f}  t={time.time() - T0:.0f}s", flush=True)
pd.DataFrame(P, columns=LABELS).assign(id=train.id.values[va]).to_csv(f"{OUT}/holdout_logprobs.csv", index=False)
del model; torch.cuda.empty_cache()

model = fit(texts, y_all)
model.save_pretrained(f"{OUT}/lora_full")
json.dump({"labels": LABELS, "n_layers": N_LAYERS, "max_len": MAX_LEN, "holdout_ba": ba, "base": BASE},
          open(f"{OUT}/lora_full/meta.json", "w"), indent=1)
print("done", f"t={time.time() - T0:.0f}s")
