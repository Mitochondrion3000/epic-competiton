import numpy as np, pandas as pd
from scipy.ndimage import maximum_filter1d
P="/home/user/epic_out/kaggle_nem6/jaNemVect1.1.simple_model_final.predictions.txt.gz"
T="/home/user/epic_data/jaNemVect1.1/ground_truth/csRNA-test.txt.gz"
p=pd.read_csv(P,header=None,dtype=np.float32)[0].values
y=pd.read_csv(T,header=None,dtype=np.float32)[0].values
print("позиций",len(p),"стартов",(y>0).sum())
s=y>0; N=s.sum()
order=np.argsort(-p,kind="stable")
for k in (0,2,5,20,50):
    near=maximum_filter1d(s.astype(np.uint8),2*k+1)>0 if k else s
    top=order[:N]
    print(f"топ-N: доля в пределах ±{k}: {near[top].mean():.3f}")
for q in (0.005,0.01,0.03,0.1):
    thr=np.quantile(p,1-q); sel=maximum_filter1d((p>=thr).astype(np.uint8),101)>0
    print(f"окна ±50 вокруг топ-{q:.1%}: доля генома {sel.mean():.3f}, стартов {sel[s].mean():.3f}, счёта {y[sel].sum()/y.sum():.3f}, доля стартов внутри окон {s[sel].mean():.4f}")
# внутри окон: хорошо ли выбрана буква? для каждого старта ранг его счёта среди позиций ±10
