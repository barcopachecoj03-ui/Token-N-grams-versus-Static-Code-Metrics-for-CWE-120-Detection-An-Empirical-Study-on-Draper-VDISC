"""
Stratified random subsample (by CWE-120 label) of the official Draper VDISC partitions, for machines with little memory.
Input : DATA_DIR with files__2_.zip (or VDISC_train_parte*.csv), VDISC_validate.csv, VDISC_test.csv
Output: CSV_PATH 
"""
import os, glob, zipfile
import pandas as pd

DATA_DIR = os.environ.get("DATA_DIR", ".")
OUT = os.environ.get("CSV_PATH", "vdisc_reduced.csv")
N = {"train": 100_000, "validate": 30_000, "test": 30_000}
TOTAL = {"train": 1_019_471, "validate": 127_476, "test": 127_419}
SEED = 42
LAB = ["CWE-119", "CWE-120", "CWE-469", "CWE-476", "CWE-other"]

def sample(df, part):
    f = N[part] / TOTAL[part]
    out = df.groupby("CWE-120", group_keys=False).sample(frac=f, random_state=SEED)   # keeps the CWE-120 prevalence
    out = out.copy(); out["split"] = part
    return out

chunks = []
parts = sorted(glob.glob(os.path.join(DATA_DIR, "VDISC_train_parte*.csv")))
if parts:
    for p in parts:
        chunks.append(sample(pd.read_csv(p), "train"))
else:
    z = zipfile.ZipFile(glob.glob(os.path.join(DATA_DIR, "*.zip"))[0])
    for n in sorted(x for x in z.namelist() if "VDISC_train_parte" in x):
        with z.open(n) as fh:
            chunks.append(sample(pd.read_csv(fh), "train"))
for part, fn in (("validate", "VDISC_validate.csv"), ("test", "VDISC_test.csv")):
    chunks.append(sample(pd.read_csv(os.path.join(DATA_DIR, fn)), part))
df = pd.concat(chunks, ignore_index=True)
df = df[df["functionSource"].astype(str).str.strip().str.len() > 0].copy()
df[LAB] = df[LAB].astype(int)
df["code_len"] = df["functionSource"].str.len()
df = df[["functionSource"] + LAB + ["split", "code_len"]]
print(df.groupby("split")["CWE-120"].agg(["count", "sum", "mean"]))
df.to_csv(OUT, index=False); print("saved", OUT, df.shape)
