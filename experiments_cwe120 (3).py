"""
CWE-120 detection on Draper VDISC (v2, aligned with the user's Clean.py metrics).
M1  = 21 static metrics (10 structural + 11 risky-call counts) | M1s = 10 structural only (supplementary)
M2  = TF-IDF token n-grams (K=20,000)                          | M3  = M1 + M2
Input : vdisc_clean.csv  (columns: functionSource, CWE-119, CWE-120, CWE-469, CWE-476, CWE-other, split)
Output: results/*.csv, results/results.json, results/pr_curves.png, results/test_scores.npz
Run   : python experiments_cwe120.py
"""
import re, json, os, time
import numpy as np, pandas as pd, scipy.sparse as sp
from multiprocessing import Pool
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (average_precision_score, roc_auc_score, precision_recall_curve,
                             precision_score, recall_score, f1_score, matthews_corrcoef)
from scipy.stats import binomtest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV_PATH = os.environ.get("CSV_PATH", "vdisc_clean.csv")
CODE_COL, TARGET, SEED = "functionSource", "CWE-120", 42
K, MIN_DF, NGRAM = 20_000, 5, (1, 3)
FIT_SAMPLE = int(os.environ.get("FIT_SAMPLE", 100_000))   # train rows used to learn vocabulary/IDF (0 = all)
N_BOOT = int(os.environ.get("N_BOOT", 1000))
RUN_EXTRA = True                                           # gradient boosting on M1/M1s (robustness)
C_GRID, CW_GRID = [0.1, 1, 10], [None, "balanced"]
os.makedirs("results", exist_ok=True)

# ---------------------------------------------------------------- static metrics (Clean.py)
KEYWORDS = ["if", "else", "for", "while", "switch", "case", "&&", "||", "?", "goto"]
RISKY_CALLS = ["strcpy", "strcat", "sprintf", "gets", "memcpy", "memmove",
               "scanf", "malloc", "free", "strncpy", "alloca"]
STRUCT = ["loc", "sloc", "comment_lines", "n_tokens", "cyclomatic", "max_nesting",
          "n_pointers", "n_arrays", "n_calls", "avg_line_len"]
CALLS = [f"call_{f}" for f in RISKY_CALLS]
TOK = re.compile(r"[A-Za-z_]\w*|\d+|==|!=|<=|>=|&&|\|\||->|\+\+|--|[^\s\w]")

def static_metrics(code):
    lines = code.split("\n")
    non_empty = [l for l in lines if l.strip()]
    comments = sum(1 for l in non_empty if l.strip().startswith(("//", "/*", "*")))
    tokens = TOK.findall(code)
    cc = 1 + sum(code.count(k) if not k.isalpha() else len(re.findall(rf"\b{k}\b", code)) for k in KEYWORDS)
    depth = max_depth = 0
    for ch in code:
        if ch == "{":
            depth += 1; max_depth = max(max_depth, depth)
        elif ch == "}":
            depth = max(depth - 1, 0)
    m = [len(lines), len(non_empty), comments, len(tokens), cc, max_depth, code.count("*"),
         code.count("["), len(re.findall(r"\b\w+\s*\(", code)),
         float(np.mean([len(l) for l in non_empty])) if non_empty else 0.0]
    m += [len(re.findall(rf"\b{fn}\s*\(", code)) for fn in RISKY_CALLS]
    return m

# ---------------------------------------------------------------- n-gram preprocessing
COMMENT_RE = re.compile(r"//[^\n]*|/\*.*?\*/", re.S)
STR_RE = re.compile(r'"(?:\\.|[^"\\\n])*"')
def prep(s):
    return STR_RE.sub(" ", COMMENT_RE.sub(" ", s))

def main():
    t0 = time.time()
    import sys, sklearn
    open("results/versions.txt", "w").write(
        f"python={sys.version.split()[0]}\nsklearn={sklearn.__version__}\nnumpy={np.__version__}\n"
        f"pandas={pd.__version__}\nscipy={__import__('scipy').__version__}\n")
    import gc, shutil
    from types import SimpleNamespace
    cdir = f"cache_k{K}_fit{FIT_SAMPLE}"            # features checkpoint
    os.makedirs(cdir, exist_ok=True)
    cached = all(os.path.exists(f"{cdir}/{f}") for f in ("M.npy", "X2_tr.npz", "X2_va.npz", "X2_te.npz", "vocab.json", "desc.csv"))
    df = pd.read_csv(CSV_PATH, usecols=[TARGET, "CWE-119", "split"] if cached else None)
    y = df[TARGET].values.astype(int)
    sp_col = df["split"].astype(str).str.lower()
    tr, va, te = (sp_col == "train").values, sp_col.isin(["validate", "validation", "val"]).values, (sp_col == "test").values
    print("sizes", tr.sum(), va.sum(), te.sum(), "| prevalence", y[tr].mean().round(4), y[va].mean().round(4), y[te].mean().round(4))
    print("share of CWE-120 positives also CWE-119:", df.loc[df[TARGET] == 1, "CWE-119"].astype(int).mean().round(3))
    rng = np.random.default_rng(SEED)
    idx = {"tr": np.where(tr)[0], "va": np.where(va)[0], "te": np.where(te)[0]}

    if cached:
        print("loading cached features from", cdir, flush=True)
        M = np.load(f"{cdir}/M.npy")
        X2 = {k: sp.load_npz(f"{cdir}/X2_{k}.npz") for k in idx}
        ng = json.load(open(f"{cdir}/vocab.json"))
        shutil.copy(f"{cdir}/desc.csv", "results/metrics_descriptives_train.csv")
        del df
    else:
        # ---- M1 / M1s (metrics computed lazily, text never copied into a list)
        ser = df[CODE_COL]
        with Pool() as p:
            M = np.array(list(p.imap(static_metrics, ser, chunksize=5000)), dtype=np.float64)
        desc = pd.DataFrame(M[tr], columns=STRUCT + CALLS).describe().T.round(4)
        desc.to_csv("results/metrics_descriptives_train.csv"); desc.to_csv(f"{cdir}/desc.csv")
        M = np.log1p(M)                                   # heavy-tailed counts -> log scale
        M = np.clip(StandardScaler().fit(M[tr]).transform(M), -5, 5).astype(np.float32)   # train-only statistics
        np.save(f"{cdir}/M.npy", M)
        print("metrics done", round(time.time() - t0), "s", flush=True)

        # ---- M2: vocabulary/IDF on (a sample of) train, then transform each partition separately
        vec = TfidfVectorizer(preprocessor=prep, token_pattern=TOK.pattern, lowercase=False, ngram_range=NGRAM,
                              max_features=K, min_df=MIN_DF, sublinear_tf=True, dtype=np.float32)
        fit_idx = idx["tr"] if FIT_SAMPLE in (0, None) or FIT_SAMPLE >= len(idx["tr"]) else rng.choice(idx["tr"], FIT_SAMPLE, replace=False)
        vec.fit(ser.iloc[fit_idx])
        print("tf-idf fitted", round(time.time() - t0), "s", flush=True)
        def transform(ix, chunk=100_000):
            blocks = [vec.transform(ser.iloc[ix[i:i + chunk]]) for i in range(0, len(ix), chunk)]
            out = sp.vstack(blocks, format="csr") if len(blocks) > 1 else blocks[0]
            del blocks; return out
        X2 = {}
        for k, ix in idx.items():
            X2[k] = transform(ix)
            sp.save_npz(f"{cdir}/X2_{k}.npz", X2[k], compressed=False)
        ng = list(vec.get_feature_names_out())
        json.dump(ng, open(f"{cdir}/vocab.json", "w"))
        print("tf-idf transformed", round(time.time() - t0), "s", X2["tr"].shape, flush=True)
        del df, ser                                      
    gc.collect()
    ytr, yva, yt = y[idx["tr"]], y[idx["va"]], y[idx["te"]]
    X1 = {k: sp.csr_matrix(M[ix]) for k, ix in idx.items()}
    X1s = {k: sp.csr_matrix(M[ix][:, :10]) for k, ix in idx.items()}
    names = {"M1": STRUCT + CALLS, "M1s": STRUCT, "M2": ng, "M3": ng + STRUCT + CALLS}
    getX = {"M1": lambda: X1, "M2": lambda: X2, "M1s": lambda: X1s,
            "M3": lambda: {k: sp.hstack([X2[k], X1[k]], format="csr") for k in idx}}

    # ---- fit, select C/class-weight on validation PR-AUC, threshold on validation F1
    scores, preds, best, models = {}, {}, {}, {}
    for cfg in ("M1", "M2", "M3", "M1s"):
        ck, ckj = f"{cdir}/cfg_{cfg}.npz", f"{cdir}/cfg_{cfg}.json"
        if os.path.exists(ck) and os.path.exists(ckj):
            z = np.load(ck); best[cfg] = json.load(open(ckj)); scores[cfg] = z["scores"]
            preds[cfg] = (z["scores"] >= best[cfg]["threshold"]).astype(int); models[cfg] = SimpleNamespace(coef_=z["coef"])
            print(cfg, "loaded from checkpoint", flush=True); continue
        X, top = None, None
        for cw in CW_GRID:
            for C in C_GRID:
                gk = f"{cdir}/fit_{cfg}_{C}_{cw}.npz"        # one checkpoint per grid point
                if os.path.exists(gk):
                    z = np.load(gk); sv, st, coef = z["sv"], z["st"], z["coef"]
                else:
                    if X is None:
                        X = getX[cfg]()
                    m = LogisticRegression(C=C, class_weight=cw, solver="lbfgs", max_iter=300).fit(X["tr"], ytr)
                    sv, st, coef = m.decision_function(X["va"]), m.decision_function(X["te"]), m.coef_
                    np.savez(gk, sv=sv, st=st, coef=coef)
                ap = average_precision_score(yva, sv)
                print(cfg, "C", C, "cw", cw, "val PR-AUC", round(ap, 4), round(time.time() - t0), "s", flush=True)
                if top is None or ap > top[0]:
                    top = (ap, C, cw, sv, st, coef)
        ap, C, cw, sv, s_, coef = top
        pr, rc, th = precision_recall_curve(yva, sv)
        f1v = 2 * pr[:-1] * rc[:-1] / np.maximum(pr[:-1] + rc[:-1], 1e-12)
        thr = float(th[np.argmax(f1v)])
        best[cfg] = dict(C=C, class_weight=str(cw), val_pr_auc=float(ap), threshold=thr, val_f1=float(f1v.max()))
        scores[cfg], preds[cfg], models[cfg] = s_, (s_ >= thr).astype(int), SimpleNamespace(coef_=coef)
        np.savez(ck, scores=s_, coef=coef); json.dump(best[cfg], open(ckj, "w"))
        del X; gc.collect()
    feats = {c: None for c in ("M1", "M2", "M3", "M1s")}
    np.savez("results/test_scores.npz", y=yt, **scores)

    # ---- test metrics (test set used once)
    rows = [dict(config=c, precision=precision_score(yt, preds[c], zero_division=0), recall=recall_score(yt, preds[c]),
                 f1=f1_score(yt, preds[c]), mcc=matthews_corrcoef(yt, preds[c]),
                 pr_auc=average_precision_score(yt, scores[c]), roc_auc=roc_auc_score(yt, scores[c]), **best[c])
            for c in feats]
    pd.DataFrame(rows).round(4).to_csv("results/test_metrics.csv", index=False)
    print(pd.DataFrame(rows).round(4).to_string())
    print("no-skill PR-AUC (prevalence):", yt.mean().round(4))

    # ---- paired bootstrap + McNemar
    def f1_fast(a, b):
        tp = (a & b).sum(); return 2 * tp / max(2 * tp + (b & ~a).sum() + (a & ~b).sum(), 1)
    brng = np.random.default_rng(SEED + 1)
    boot = {c: {"f1": [], "pr_auc": []} for c in feats}
    n, yb = len(yt), yt.astype(bool)
    for _ in range(N_BOOT):
        bi = brng.integers(0, n, n)
        for c in feats:
            boot[c]["f1"].append(f1_fast(yb[bi], preds[c][bi].astype(bool)))
            boot[c]["pr_auc"].append(average_precision_score(yt[bi], scores[c][bi]))
    pd.DataFrame([dict(config=c, metric=k, mean=np.mean(v), lo=np.percentile(v, 2.5), hi=np.percentile(v, 97.5))
                  for c in feats for k, v in boot[c].items()]).round(4).to_csv("results/bootstrap_ci.csv", index=False)
    out = []
    for a, b, primary in [("M2", "M1", 1), ("M3", "M1", 1), ("M3", "M2", 1), ("M1", "M1s", 0), ("M2", "M1s", 0)]:
        ca, cb = preds[a] == yt, preds[b] == yt
        n10, n01 = int((ca & ~cb).sum()), int((~ca & cb).sum())
        pm = binomtest(min(n10, n01), n10 + n01, 0.5).pvalue if n10 + n01 else 1.0
        for k in ("f1", "pr_auc"):
            d = np.array(boot[a][k]) - np.array(boot[b][k])
            out.append(dict(comparison=f"{a} vs {b}", primary=primary, metric=k, diff=d.mean(),
                            lo=np.percentile(d, 2.5), hi=np.percentile(d, 97.5),
                            boot_p=min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())),
                            mcnemar_p=pm if k == "f1" else np.nan, only_first_correct=n10, only_second_correct=n01))
    pd.DataFrame(out).round(4).to_csv("results/pairwise_tests.csv", index=False)
    print(pd.DataFrame(out).round(4).to_string()); print("Bonferroni alpha (3 primary comparisons) =", round(0.05 / 3, 4))

    # ---- coefficients (and rank among all n-grams) of selected tokens, to check the interpretability claims
    w2 = models["M2"].coef_.ravel(); vocab = {t: i for i, t in enumerate(names["M2"])}
    watch = RISKY_CALLS + ["sizeof", "strlen", "strnlen", "snprintf", "strncat", "fgets", "NULL"]
    pd.DataFrame([dict(token=t, coef=w2[vocab[t]], rank_from_top=int((w2 > w2[vocab[t]]).sum()) + 1)
                  for t in watch if t in vocab]).round(3).to_csv("results/watch_tokens_M2.csv", index=False)

    # ---- influential features
    for cfg in ("M2", "M3"):
        w = models[cfg].coef_.ravel(); nm = np.array(names[cfg]); o = np.argsort(w)
        pd.DataFrame({"top_positive": nm[o[::-1][:25]], "coef_pos": w[o[::-1][:25]],
                      "top_negative": nm[o[:25]], "coef_neg": w[o[:25]]}).round(3).to_csv(f"results/top_features_{cfg}.csv", index=False)
    for cfg in ("M1", "M1s"):
        print(cfg, "coefficients:", dict(zip(names[cfg], models[cfg].coef_.ravel().round(3))))

    # ---- robustness: non-linear model on the dense metric sets
    if RUN_EXTRA:
        ex = []
        for cfg in ("M1", "M1s"):
            cols = slice(0, 21) if cfg == "M1" else slice(0, 10)
            Md = {k: M[ix][:, cols] for k, ix in idx.items()}
            gb = HistGradientBoostingClassifier(random_state=SEED).fit(Md["tr"], ytr)
            pr, rc, th = precision_recall_curve(yva, gb.predict_proba(Md["va"])[:, 1])
            thr = th[np.argmax(2 * pr[:-1] * rc[:-1] / np.maximum(pr[:-1] + rc[:-1], 1e-12))]
            s = gb.predict_proba(Md["te"])[:, 1]; p_ = (s >= thr).astype(int)
            ex.append(dict(config=cfg + "-GB", precision=precision_score(yt, p_, zero_division=0), recall=recall_score(yt, p_),
                           f1=f1_score(yt, p_), pr_auc=average_precision_score(yt, s), roc_auc=roc_auc_score(yt, s)))
        pd.DataFrame(ex).round(4).to_csv("results/extra_models.csv", index=False); print(pd.DataFrame(ex).round(4))

    plt.figure(figsize=(5, 4))
    for c in feats:
        p_, r_, _ = precision_recall_curve(yt, scores[c]); plt.plot(r_, p_, label=c)
    plt.axhline(yt.mean(), ls="--", c="gray", label="no-skill")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.legend(); plt.tight_layout(); plt.savefig("results/pr_curves.png", dpi=200)
    json.dump(best, open("results/results.json", "w"), indent=2)
    print("total", round(time.time() - t0), "s")

if __name__ == "__main__":
    main()
