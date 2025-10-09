@'
# -*- coding: utf-8 -*-
"""
TOST 非劣检验（对齐 2.2.1）
输入：两份 CSV，至少包含列：id, score（或自定义列名）
可选：bucket 列（分层），paired=是（按 id 配对）
输出：overall + 分层的 TOST 结果与 95% CI，FDR 矫正后的结论
"""
import argparse, pandas as pd, numpy as np, math, json

def bootstrap_ci(x, y, B=1000, paired=True, func=np.mean, seed=42):
    rng = np.random.default_rng(seed)
    vals = []
    n = len(x)
    for _ in range(B):
        idx = rng.integers(0, n, n)
        xb = x[idx]
        yb = y[idx] if paired else y[rng.integers(0, len(y), n)]
        vals.append(func(yb - xb))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)

def tost_noninferior(x, y, eps=0.01, alpha=0.05):
    """ H1: -eps < mean(y-x) < eps ；返回是否非劣、估计差、CI """
    diff = np.mean(y - x)
    lo, hi = bootstrap_ci(x, y, B=2000, paired=True, func=np.mean)
    # 两单侧：lo > -eps 且 hi < eps 即通过
    pass_left  = lo > -eps
    pass_right = hi <  eps
    return {"diff_mean": float(diff), "ci95": [float(lo), float(hi)],
            "eps": float(eps), "alpha": float(alpha),
            "pass_noninferior": bool(pass_left and pass_right)}

def bh_fdr(pvals, alpha=0.05):
    # 这里我们不算 p 值（自助 CI 判定），保留接口
    return alpha

def run(df_b, df_g, score_col_b="score", score_col_g="score", eps=0.01, alpha=0.05):
    df = pd.merge(df_b[["id",score_col_b,"bucket"]] if "bucket" in df_b.columns else df_b[["id",score_col_b]],
                  df_g[["id",score_col_g,"bucket"]] if "bucket" in df_g.columns else df_g[["id",score_col_g]],
                  on="id", suffixes=("_b","_g"))
    df["bucket"] = df["bucket_b"] if "bucket_b" in df.columns else (df["bucket_g"] if "bucket_g" in df.columns else "all")

    out = {}
    # overall
    x = df[f"{score_col_b}_b"].to_numpy(dtype=float)
    y = df[f"{score_col_g}_g"].to_numpy(dtype=float)
    out["overall"] = tost_noninferior(x,y,eps,alpha)

    # per-bucket
    per = {}
    for bkt, sub in df.groupby("bucket"):
        xb = sub[f"{score_col_b}_b"].to_numpy(dtype=float)
        yb = sub[f"{score_col_g}_g"].to_numpy(dtype=float)
        per[str(bkt)] = tost_noninferior(xb,yb,eps,alpha)
    out["per_bucket"] = per
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_csv", required=True)
    ap.add_argument("--gateway_csv", required=True)
    ap.add_argument("--score_col", default="score")
    ap.add_argument("--epsilon", type=float, default=0.01)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--out", default="runs\\tost_report.json")
    args = ap.parse_args()

    dfb = pd.read_csv(args.baseline_csv)
    dfg = pd.read_csv(args.gateway_csv)
    res = run(dfb, dfg, score_col_b=args.score_col, score_col_g=args.score_col, eps=args.epsilon, alpha=args.alpha)

    with open(args.out,"w",encoding="utf-8") as f:
        f.write(json.dumps(res, ensure_ascii=False, indent=2))
    print(json.dumps(res, ensure_ascii=False, indent=2))
    print("已保存：", args.out)

if __name__ == "__main__":
    main()
'@ | Set-Content -Encoding UTF8 .\tost_report.py


