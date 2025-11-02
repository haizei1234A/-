# fix_stratum.py
import sys, pandas as pd

in_path, out_path = sys.argv[1], sys.argv[2]
df = pd.read_csv(in_path, encoding="utf-8")
if "stratum" in df.columns:
    # 只保留管道前第一段，如 "general|ctxS|light|compL" -> "general"
    df["stratum"] = df["stratum"].astype(str).str.split("|").str[0].str.strip().str.lower()
# 确保列名规范
if "metric" in df.columns:
    df["metric"] = df["metric"].astype(str).str.lower()
df.to_csv(out_path, index=False, encoding="utf-8")
print(f"OK -> {out_path}, n={len(df)}")
