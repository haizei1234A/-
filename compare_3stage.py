# -*- coding: utf-8 -*-
"""
compare_3stage.py —— 健康检查 / 看板快照（§2.2 对齐版）

对齐要点：
1) 同时展示 Raw / CO-corrected / Overall / Warm-only 四口径中的关键值；
2) 以 **CO-corrected Warm P95** 为护栏判定基准（≤ 基线 * (1+10%)）；
3) 输出测量指纹：co_correction、tdigest_compression、warmup_window_ms；
4) 兼容 bench 行级字段变更：升级率使用 (upgrades>0) 或 upgraded；
5) 同步给出 r_costly、JSON 失败率、重试/拒答率、EDP（≈ avg_eval_ms × P50）。

输入：自动抓取 runs/ 下最近一次 baseline 与 gateway 的 *.jsonl 与 *_summary.json
输出：控制台表格 + runs/compare_latest.{csv,json,txt}
"""

import os, glob, json, math
import pandas as pd

RUNS = "runs"

def latest(pat: str):
    cand = sorted(glob.glob(pat), key=os.path.getmtime, reverse=True)
    return cand[0] if cand else None

def read_latest_pair():
    # 日志（行级）
    b_jsonl = latest(os.path.join(RUNS, "baseline_enh_*.jsonl"))
    g_jsonl = latest(os.path.join(RUNS, "gateway_3stage_*.jsonl"))
    if not b_jsonl or not g_jsonl:
        raise FileNotFoundError("找不到 baseline 或 gateway 的 jsonl，先把两套都跑一遍。")
    b = pd.read_json(b_jsonl, lines=True)
    g = pd.read_json(g_jsonl, lines=True)

    # Summary（优先；用于 CO 指标与指纹）
    b_sum = latest(os.path.join(RUNS, "baseline_enh_*_summary.json"))
    g_sum = latest(os.path.join(RUNS, "gateway_3stage_*_summary.json"))
    sb = json.load(open(b_sum, "r", encoding="utf-8-sig")) if b_sum else {}
    sg = json.load(open(g_sum, "r", encoding="utf-8-sig")) if g_sum else {}

    return (b, sb, b_jsonl), (g, sg, g_jsonl)

def pct(a, b):
    return 100.0 * (a - b) / max(b, 1e-9)

def rate(series_bool_like) -> float:
    if series_bool_like is None or len(series_bool_like) == 0:
        return float("nan")
    s = series_bool_like.astype("bool")
    return float(100.0 * s.mean())

def warm_only(df: pd.DataFrame) -> pd.DataFrame:
    if "in_warmup" in df.columns:
        return df[~df["in_warmup"]].copy()
    return df.copy()

def infer_upgraded_rate(df: pd.DataFrame) -> float:
    if "upgrades" in df.columns:
        return rate(df["upgrades"].fillna(0).astype("int") > 0)
    if "upgraded" in df.columns:
        return rate(df["upgraded"])
    return float("nan")

def summarize_from_logs(df: pd.DataFrame) -> dict:
    """在没有 summary 的情况下，回退基于日志的 Raw 指标（不含 CO 指标）。"""
    w = warm_only(df)
    dist = w["final_path"].value_counts(normalize=True) * 100.0 if "final_path" in w.columns else pd.Series()
    need_json = w["need_json"] if "need_json" in w.columns else None
    json_ok = w["json_ok"] if "json_ok" in w.columns else None

    json_fail_rate = float("nan")
    if need_json is not None:
        mask = need_json.astype(bool)
        if mask.any() and (json_ok is not None):
            json_fail_rate = float(100.0 * (~json_ok[mask].astype(bool)).mean())

    return dict(
        warm_p50_ms=float(w["e2e_ms"].quantile(0.5)) if "e2e_ms" in w.columns else float("nan"),
        warm_p95_ms=float(w["e2e_ms"].quantile(0.95)) if "e2e_ms" in w.columns else float("nan"),
        overall_p50_ms=float(df["e2e_ms"].quantile(0.5)) if "e2e_ms" in df.columns else float("nan"),
        overall_p95_ms=float(df["e2e_ms"].quantile(0.95)) if "e2e_ms" in df.columns else float("nan"),
        avg_eval_ms=float(w["e2e_ms"].mean()) if "e2e_ms" in w.columns else float("nan"),
        sum_eval_ms=float(w["e2e_ms"].sum()) if "e2e_ms" in w.columns else float("nan"),
        light_pct=float(dist.get("light", 0.0)),
        std_pct=float(dist.get("std", 0.0)),
        enh_pct=float(dist.get("enh", 0.0)),
        r_costly_pct=float((w["final_path"] == "enh").mean() * 100.0) if "final_path" in w.columns else float("nan"),
        json_fail_rate_pct=json_fail_rate,
        repaired_rate_pct=rate(w["repaired"]) if "repaired" in w.columns else float("nan"),
        upgraded_rate_pct=infer_upgraded_rate(w),
        retry_rate_pct=rate(w["retry_count"] > 0) if "retry_count" in w.columns else float("nan"),
        refusal_rate_pct=rate(w["refusal"]) if "refusal" in w.columns else float("nan"),
        # 指纹未知（在无 summary 情况下留空）
        co_correction=None, tdigest_compression=None, warmup_window_ms=None,
    )

def attach_from_summary(base: dict, summ: dict) -> dict:
    """用 summary 覆盖/补全关键字段（含 CO 指标与指纹）。"""
    out = dict(base)
    # 四口径（若 summary 提供则覆盖）
    for k in ["warm_p50_ms","warm_p95_ms","overall_p50_ms","overall_p95_ms","avg_eval_ms","sum_eval_ms",
              "light_pct","std_pct","enh_pct","r_costly_pct","json_fail_rate","json_fail_rate_pct",
              "retry_rate_pct","refusal_rate_pct"]:
        if k in summ and summ.get(k) is not None:
            out[k] = float(summ[k])

    # CO 指标与指纹
    if "warm_p95_ms_co" in summ:
        out["warm_p95_ms_co"] = float(summ["warm_p95_ms_co"])
    else:
        out["warm_p95_ms_co"] = float("nan")

    out["co_correction"] = summ.get("co_correction", out.get("co_correction"))
    out["tdigest_compression"] = summ.get("tdigest_compression", out.get("tdigest_compression"))
    out["warmup_window_ms"] = summ.get("warmup_window_ms", out.get("warmup_window_ms"))
    return out

def build_row(df: pd.DataFrame, summ: dict, label: str) -> dict:
    base = summarize_from_logs(df)
    row = attach_from_summary(base, summ or {})
    row["mode"] = label
    # 计算 EDP ≈ avg_eval_ms × warm_p50_ms
    row["EDP_approx"] = float(row.get("avg_eval_ms", float("nan")) * row.get("warm_p50_ms", float("nan")))
    # 统一 json_fail_rate_pct 字段（summary 可能给的是 0~1）
    if "json_fail_rate_pct" in row and (row["json_fail_rate_pct"] is None or math.isnan(row["json_fail_rate_pct"])):
        # 若 summary 给了 0~1 的 json_fail_rate
        j = summ.get("json_fail_rate", None)
        if j is not None:
            row["json_fail_rate_pct"] = float(j) * 100.0
    return row

def main():
    (b_df, b_sum, b_path), (g_df, g_sum, g_path) = read_latest_pair()
    b_w = build_row(b_df, b_sum, "baseline")
    g_w = build_row(g_df, g_sum, "gateway")

    # Δ 与护栏（以 CO-corrected Warm P95 为准；若缺 CO 则回退 Raw）
    b_p95 = b_w.get("warm_p95_ms_co", float("nan"))
    g_p95 = g_w.get("warm_p95_ms_co", float("nan"))
    fallback_used = False
    if (b_p95 is None or math.isnan(b_p95)) or (g_p95 is None or math.isnan(g_p95)):
        b_p95 = b_w.get("warm_p95_ms", float("nan"))
        g_p95 = g_w.get("warm_p95_ms", float("nan"))
        fallback_used = True
    delta_p95 = pct(g_p95, b_p95) if (not math.isnan(b_p95) and not math.isnan(g_p95)) else float("nan")
    guard_pass = (delta_p95 <= 10.0) if not math.isnan(delta_p95) else False

    delta_p50 = pct(g_w.get("warm_p50_ms", float("nan")), b_w.get("warm_p50_ms", float("nan")))
    delta_avg_eval = pct(g_w.get("avg_eval_ms", float("nan")), b_w.get("avg_eval_ms", float("nan")))

    # 打印
    hdr = [
        "mode","n_ok","warm_P50(ms)","warm_P95(ms)","warm_P95_CO(ms)",
        "avg_eval_ms","sum_eval_ms","light(%)","std(%)","enh(%)",
        "r_costly(%)","json_fail(%)","retry(%)","refusal(%)",
        "repaired(%)","upgraded(%)","EDP"
    ]
    def fmt(row):
        return (
            row["mode"],
            int((~pd.isna(warm_only(b_df if row["mode"]=="baseline" else g_df)["e2e_ms"])).sum()),
            f'{row.get("warm_p50_ms", float("nan")):9.1f}',
            f'{row.get("warm_p95_ms", float("nan")):9.1f}',
            f'{row.get("warm_p95_ms_co", float("nan")):11.1f}',
            f'{row.get("avg_eval_ms", float("nan")):11.1f}',
            f'{row.get("sum_eval_ms", float("nan")):12.1f}',
            f'{row.get("light_pct", float("nan")):7.1f}',
            f'{row.get("std_pct", float("nan")):6.1f}',
            f'{row.get("enh_pct", float("nan")):7.1f}',
            f'{row.get("r_costly_pct", float("nan")):10.1f}',
            f'{row.get("json_fail_rate_pct", float("nan")):10.2f}',
            f'{row.get("retry_rate_pct", float("nan")):7.2f}',
            f'{row.get("refusal_rate_pct", float("nan")):8.2f}',
            f'{row.get("repaired_rate_pct", float("nan")):11.2f}',
            f'{row.get("upgraded_rate_pct", float("nan")):11.2f}',
            f'{row.get("EDP_approx", float("nan")):9.0f}',
        )

    print("".join([
        "mode     n_ok  warm_P50  warm_P95  warm_P95_CO   avg_eval_ms   sum_eval_ms  light%  std%  enh%  r_costly%  json_fail%  retry%  refusal%  repaired%  upgraded%       EDP\n",
        "-"*132,"\n",
        f"{fmt(b_w)[0]:8s} {fmt(b_w)[1]:5d} {fmt(b_w)[2]:>9s} {fmt(b_w)[3]:>9s} {fmt(b_w)[4]:>11s} {fmt(b_w)[5]:>12s} {fmt(b_w)[6]:>12s} {fmt(b_w)[7]:>7s} {fmt(b_w)[8]:>6s} {fmt(b_w)[9]:>7s} {fmt(b_w)[10]:>10s} {fmt(b_w)[11]:>10s} {fmt(b_w)[12]:>7s} {fmt(b_w)[13]:>8s} {fmt(b_w)[14]:>11s} {fmt(b_w)[15]:>11s} {fmt(b_w)[16]:>9s}\n",
        f"{fmt(g_w)[0]:8s} {fmt(g_w)[1]:5d} {fmt(g_w)[2]:>9s} {fmt(g_w)[3]:>9s} {fmt(g_w)[4]:>11s} {fmt(g_w)[5]:>12s} {fmt(g_w)[6]:>12s} {fmt(g_w)[7]:>7s} {fmt(g_w)[8]:>6s} {fmt(g_w)[9]:>7s} {fmt(g_w)[10]:>10s} {fmt(g_w)[11]:>10s} {fmt(g_w)[12]:>7s} {fmt(g_w)[13]:>8s} {fmt(g_w)[14]:>11s} {fmt(g_w)[15]:>11s} {fmt(g_w)[16]:>9s}\n",
    ]))

    print(f"ΔP50% = {delta_p50:6.2f}    ΔP95%(CO{'-fallback-RAW' if fallback_used else ''}) = {delta_p95:6.2f}    Δavg_eval_ms% = {delta_avg_eval:6.2f}")
    if not fallback_used:
        print(f"[Guardrail·CO] P95_CO(gateway) ≤ 1.10 × P95_CO(baseline) ?  ==>  {'PASS' if guard_pass else 'FAIL'}")
    else:
        print("[Guardrail] 未找到 CO 指标，已回退 Raw P95 判定（请检查 bench 是否已输出 warm_p95_ms_co）")

    # —— 落盘：CSV / JSON / TXT —— 
    out_df = pd.DataFrame([b_w, g_w])
    os.makedirs(RUNS, exist_ok=True)
    out_csv = os.path.join(RUNS, "compare_latest.csv")
    out_json = os.path.join(RUNS, "compare_latest.json")
    out_txt = os.path.join(RUNS, "compare_latest.txt")
    out_df.to_csv(out_csv, index=False, encoding="utf-8")
    with open(out_json, "w", encoding="utf-8") as f:
        f.write(out_df.to_json(orient="records", force_ascii=False, indent=2))
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(json.dumps({
            "delta_p50_pct": round(float(delta_p50), 2),
            "delta_p95_pct": round(float(delta_p95), 2),
            "guardrail_pass_co": bool(guard_pass),
            "co_fallback_used": bool(fallback_used),
            "baseline_fingerprint": {
                "co_correction": b_w.get("co_correction"),
                "tdigest_compression": b_w.get("tdigest_compression"),
                "warmup_window_ms": b_w.get("warmup_window_ms")
            },
            "gateway_fingerprint": {
                "co_correction": g_w.get("co_correction"),
                "tdigest_compression": g_w.get("tdigest_compression"),
                "warmup_window_ms": g_w.get("warmup_window_ms")
            }
        }, ensure_ascii=False))
    print(f"结果已保存: {out_csv} | {out_json} | {out_txt}")

if __name__ == "__main__":
    main()
