import sys, numpy as np, pandas as pd, modiscolite
sp, path = sys.argv[1], sys.argv[2]
d = np.load(path); H = d["hyp"]; X = d["onehot"].astype(np.float32); CROP = H.shape[1] // 2
pos, neg = modiscolite.tfmodisco.TFMoDISco(hypothetical_contribs=H, one_hot=X, sliding_window_size=15, flank_size=5,
    target_seqlet_fdr=0.05, max_seqlets_per_metacluster=30000, n_leiden_runs=5, verbose=False)
modiscolite.io.save_hdf5(f"/tmp/mod/{sp}.h5", pos, neg, 15)
ACG = np.array(list("ACGT")); res = []
for sign, pats in (("pos", pos), ("neg", neg)):
    for k, p in enumerate(pats or []):
        ppm = np.array(p["sequence"]); sl = p["seqlets"]; n = len(sl["start"])
        cen = ((np.array(sl["start"]) + np.array(sl["end"])) / 2 - CROP)
        ic = float((ppm * np.log2(np.clip(ppm, 1e-9, None) / 0.25)).sum())
        res.append(dict(species=sp, pattern=f"{sign}_{k}", seqlets=n, consensus="".join(ACG[ppm.argmax(1)]), ic=round(ic,1),
                        offset_med=round(float(np.median(cen)),1), offset_sd=round(float(cen.std()),1)))
        np.save(f"/tmp/mod/{sp}_{sign}_{k}_ppm.npy", ppm)
pd.DataFrame(res).to_csv(f"/tmp/mod/{sp}_patterns.csv", index=False); print(pd.DataFrame(res).to_string())
