"""Step 2. Satellite canopy methods in the independent field test (leave-one-field-out).

For each held-out field k, everything is chosen on the two training fields only:
  * the co-registration shift between PlanetScope and the UAV grid (max NDVI-fraction correlation)
  * NDVI / NDRE thresholds (spectral baselines)
  * the decision threshold of each classifier (inner leave-one-field-out on the training fields)
Then field k is predicted once.

Methods
  * NDVI and NDRE thresholds
  * 1 date (12 Jan): random forest, extra trees, gradient boosting (HGB), logistic regression, MLP
  * 3 dates (12, 15, 22 Jan): random forest and HGB
  * label rules: 3 m pixel = tree if UAV fraction >= 0.5 (main) or >= 0.2 (sensitivity)
Features (139 per date): 8 bands, all 28 normalized differences, EVI, and local mean / SD /
deviation (3-21 px windows), range and gradient of NDVI, NDRE, NIR, red edge, red and green.

Outputs: outputs/satellite_tests.json, outputs/threshold_sweep_3dateRF.json, outputs/Figure4b_threshold.png
"""
import itertools
import json
import time

import numpy as np
from scipy import ndimage
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, jaccard_score, precision_score, recall_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from utils import best_shift, best_threshold, load_config, load_scene, nd, out, p, rp, save_json, shift_bands

FIELDS = (1, 2, 3)


def stats(v, sizes=(3, 5, 9, 15, 21)):
    o, n = [], []
    for s in sizes:
        m = ndimage.uniform_filter(v, s)
        sd = np.sqrt(np.maximum(ndimage.uniform_filter(v * v, s) - m * m, 0))
        o += [m, sd, v - m]
        n += [f"m{s}", f"sd{s}", f"dev{s}"]
    o += [ndimage.maximum_filter(v, 3) - ndimage.minimum_filter(v, 3), ndimage.gaussian_gradient_magnitude(v, 1.0)]
    n += ["rng3", "grad"]
    return o, n


def features(img):
    x = img / 10000.0
    valid = (x > 0).all(0)
    F = [x[i] for i in range(8)]
    for i, j in itertools.combinations(range(8), 2):
        F.append(nd(x[j], x[i]))
    F.append(2.5 * (x[7] - x[5]) / (x[7] + 6 * x[5] - 7.5 * x[1] + 1 + 1e-6))
    for v in (nd(x[7], x[5]), nd(x[7], x[6]), x[7], x[6], x[5], x[3]):
        s, _ = stats(v)
        F += s
    return np.stack(F).astype(np.float32), valid


MODELS = {
    "Random forest": lambda s: RandomForestClassifier(n_estimators=150, min_samples_leaf=3, max_samples=0.4,
                                                      class_weight="balanced_subsample", n_jobs=-1, random_state=s),
    "Extra trees": lambda s: ExtraTreesClassifier(n_estimators=200, min_samples_leaf=3, class_weight="balanced_subsample",
                                                  n_jobs=-1, random_state=s),
    "Gradient boosting": lambda s: HistGradientBoostingClassifier(max_iter=250, learning_rate=0.05, max_leaf_nodes=31,
                                                                  l2_regularization=1.0, class_weight="balanced", random_state=s),
    "Logistic regression": lambda s: make_pipeline(StandardScaler(), LogisticRegression(C=0.5, class_weight="balanced", max_iter=400)),
    "MLP": lambda s: make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-3,
                                                                   early_stopping=True, max_iter=60, random_state=s)),
}
CONFIGS = [("1 date", [0], ["Random forest", "Extra trees", "Gradient boosting", "Logistic regression", "MLP"]),
           ("3 dates", [0, 1, 2], ["Random forest", "Gradient boosting"])]


def main():
    cfg = load_config()
    sc = cfg["satellite"]
    seed = sc["seed"]
    cands = [(dy, dx) for dy in sc["shift_candidates_dy"] for dx in sc["shift_candidates_dx"]]
    a, b, s = sc["threshold_grid"]
    TH = np.round(np.arange(a, b, s), 2)

    main_img, G = load_scene(p(cfg, "scene_main"))
    H, W = main_img.shape[1:]
    dates = [main_img] + [load_scene(rp(f), G, (H, W))[0] for f in cfg["paths"]["scenes_extra"]]
    Z = np.load(out(cfg, "reference_fractions.npz"))
    Fr = {k: Z[f"frac3_{k}"] for k in FIELDS}

    # crop to the fields (+40 px) to keep memory small
    anyf = np.zeros((H, W), bool)
    for k in FIELDS:
        anyf |= np.isfinite(Fr[k])
    r, c = np.where(anyf)
    r0, r1, c0, c1 = max(r.min() - 40, 0), min(r.max() + 41, H), max(c.min() - 40, 0), min(c.max() + 41, W)
    dates = [d[:, r0:r1, c0:c1].copy() for d in dates]
    Fr = {k: v[r0:r1, c0:c1] for k, v in Fr.items()}
    ndv = [nd(d[7], d[5]) for d in dates]
    results = []

    # ------------------------------------------------ spectral baselines
    TT = np.round(np.arange(-0.2, 0.9, 0.005), 3)
    x0 = dates[0] / 10000.0
    for name, idx in (("NDVI threshold", nd(x0[7], x0[5])), ("NDRE threshold", nd(x0[7], x0[6]))):
        f1s = []
        for k in FIELDS:
            tr = [j for j in FIELDS if j != k]
            sh = best_shift(ndv[0], Fr, tr, cands)
            v = ndimage.shift(idx, sh, order=1, mode="nearest")
            yt = np.concatenate([Fr[j][np.isfinite(Fr[j])] >= 0.5 for j in tr])
            vt = np.concatenate([v[np.isfinite(Fr[j])] for j in tr])
            t = best_threshold(vt, yt, TT)
            ok = np.isfinite(Fr[k])
            f1s.append(f1_score(Fr[k][ok] >= 0.5, v[ok] >= t))
        results.append(dict(method=name, rule=0.5, F1=[round(float(v), 3) for v in f1s], F1_mean=round(float(np.mean(f1s)), 3)))
        print(results[-1], flush=True)

    # ------------------------------------------------ classifiers
    sweep = None
    for cname, use, models in CONFIGS:
        folds = {}
        for k in FIELDS:
            tr = [j for j in FIELDS if j != k]
            Fs, valid = [], np.ones(dates[0].shape[1:], bool)
            for di in use:
                sh = best_shift(ndv[di], Fr, tr, cands)
                F, v = features(shift_bands(dates[di], *sh))
                Fs.append(F)
                valid &= v
            F = np.concatenate(Fs)
            X, yf, fld = [], [], []
            for j in FIELDS:
                ok = np.isfinite(Fr[j]) & valid
                rr, cc = np.where(ok)
                X.append(F[:, rr, cc].T)
                yf.append(Fr[j][rr, cc])
                fld.append(np.full(rr.size, j))
            folds[k] = (np.vstack(X), np.concatenate(yf), np.concatenate(fld))
        for mn in models:
            for rule in sc["label_rules"]:
                f1s, P, R, AR, save = [], [], [], [], {}
                t0 = time.time()
                for k in FIELDS:
                    X, yf, fld = folds[k]
                    y = (yf >= rule).astype(int)
                    te, trm = fld == k, fld != k
                    ftr = fld[trm]
                    pin = np.zeros(trm.sum())
                    for j in np.unique(ftr):                       # inner leave-one-field-out
                        a_ = ftr == j
                        pin[a_] = MODELS[mn](seed).fit(X[trm][~a_], y[trm][~a_]).predict_proba(X[trm][a_])[:, 1]
                    t = best_threshold(pin, y[trm], TH)
                    prob = MODELS[mn](seed).fit(X[trm], y[trm]).predict_proba(X[te])[:, 1]
                    pb = prob >= t
                    save[k] = (prob, y[te], t)
                    f1s.append(f1_score(y[te], pb))
                    P.append(precision_score(y[te], pb, zero_division=0))
                    R.append(recall_score(y[te], pb))
                    AR.append(pb.sum() / max(y[te].sum(), 1))
                results.append(dict(method=f"{mn}, {cname}", rule=rule, F1=[round(float(v), 3) for v in f1s],
                                    F1_mean=round(float(np.mean(f1s)), 3), precision=round(float(np.mean(P)), 3),
                                    recall=round(float(np.mean(R)), 3), area_ratio=round(float(np.mean(AR)), 2),
                                    seconds=round(time.time() - t0)))
                print(results[-1], flush=True)
                save_json(results, out(cfg, "satellite_tests.json"))
                if mn == "Random forest" and cname == "3 dates" and rule == 0.5:
                    sweep = save

    # ------------------------------------------------ threshold sweep (Figure 4b)
    prob = np.concatenate([sweep[k][0] for k in FIELDS])
    y = np.concatenate([sweep[k][1] for k in FIELDS])
    tsel = float(np.mean([sweep[k][2] for k in FIELDS]))
    T = np.round(np.arange(0.05, 0.951, 0.01), 2)
    npos = np.array([(prob >= t).sum() for t in T])
    Pp = np.where(npos >= 100, [precision_score(y, prob >= t, zero_division=1) for t in T], np.nan)
    Rr = [recall_score(y, prob >= t) for t in T]
    Ff = [f1_score(y, prob >= t, zero_division=0) for t in T]
    Ii = [jaccard_score(y, prob >= t, zero_division=0) for t in T]
    j = int(np.argmin(abs(T - tsel)))
    sw = dict(prevalence=float(y.mean()), peak_F1=float(max(Ff)), peak_threshold=float(T[int(np.argmax(Ff))]),
              chosen_threshold=tsel, F1_at_chosen=float(Ff[j]), precision_at_chosen=float(Pp[j]), recall_at_chosen=float(Rr[j]),
              thresholds=T, precision=Pp, recall=Rr, F1=Ff, IoU=Ii)
    save_json(sw, out(cfg, "threshold_sweep_3dateRF.json"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.5, 3.4))
    ax.plot(T, Pp, "--", label="Precision")
    ax.plot(T, Rr, ":", label="Recall")
    ax.plot(T, Ff, "-", lw=2, label="F1-score")
    ax.plot(T, Ii, "-.", label="IoU")
    ax.axvline(tsel, ls="--", c="k", lw=0.8)
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out(cfg, "Figure4b_threshold.png"), dpi=300)
    print(json.dumps({k: v for k, v in sw.items() if not isinstance(v, (list, np.ndarray))}, indent=1, default=float))


if __name__ == "__main__":
    main()
