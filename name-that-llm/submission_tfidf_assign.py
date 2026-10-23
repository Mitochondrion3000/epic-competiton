# Name That LLM: TF-IDF + логистическая регрессия + центрирование по темам + назначение «одна модель на тему».
# Работает на Kaggle без интернета, только CPU. CV (GroupKFold по prompt_id): 0.353 → 0.402 → 0.452.
import glob, os
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import AgglomerativeClustering
from scipy.optimize import linear_sum_assignment

def find(name):
    for p in [f"/kaggle/input/competitions/name-that-llm/{name}", f"/kaggle/input/name-that-llm/{name}", f"data/{name}"]:
        if os.path.exists(p):
            return p
    return glob.glob(f"/kaggle/input/**/{name}", recursive=True)[0]

train = pd.read_csv(find("train.csv"))
test = pd.read_csv(find("test.csv"))
print("train", train.shape, "test", test.shape)

# 1. стилевые признаки: n-граммы символов (с регистром и пунктуацией) + слова и знаки препинания
vecs = [TfidfVectorizer(analyzer="char", ngram_range=(1, 5), sublinear_tf=True, min_df=2, lowercase=False, max_features=300000),
        TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True, min_df=2, lowercase=False,
                        token_pattern=r"(?u)\b\w+\b|[^\w\s]")]
X_tr = sp.hstack([v.fit_transform(train.text) for v in vecs]).tocsr()
X_te = sp.hstack([v.transform(test.text) for v in vecs]).tocsr()

clf = LogisticRegression(C=20, max_iter=3000).fit(X_tr, train.model)
classes = clf.classes_
S = clf.predict_log_proba(X_te)
S = (S - S.mean()) / S.std()

# 2. восстанавливаем темы теста: ~18 ответов (по одному от каждой модели) на промпт
n_clusters = max(1, int(round(len(test) / len(classes))))
if len(test) > 1:
    T = TfidfVectorizer(sublinear_tf=True, stop_words="english").fit_transform(test.text).toarray()
    grp = AgglomerativeClustering(n_clusters=n_clusters, metric="cosine", linkage="average").fit_predict(T)
else:
    grp = np.zeros(len(test), int)
print("кластеров:", n_clusters, "| размеры:", pd.Series(grp).value_counts().describe().round(1).to_dict())

# 3. центрирование внутри темы убирает смещение «тема → модель»,
# 4. назначение: в каждой теме каждая модель встречается не больше одного раза
pred = np.empty(len(test), int)
for k in np.unique(grp):
    idx = np.where(grp == k)[0]
    Sk = S[idx] - S[idx].mean(0) if len(idx) > 1 else S[idx]
    if len(idx) <= len(classes):
        r, c = linear_sum_assignment(-Sk)
        pred[idx[r]] = c
    else:
        pred[idx] = Sk.argmax(1)

sub = pd.DataFrame({"id": test.id, "model": classes[pred]})
assert sub.id.is_unique and sub.model.isin(classes).all() and len(sub) == len(test)
sub.to_parquet("/kaggle/working/submission.parquet" if os.path.isdir("/kaggle/working") else "submission.parquet", index=False)
print(sub.model.value_counts())
