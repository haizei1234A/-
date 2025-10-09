# -*- coding: utf-8 -*-
"""
build_scores_from_logs.py — 主质量指标打分（对齐 §2.2）
变更要点：
- 默认将“无 gold”的样本从主指标中剔除（missing-policy=exclude）
- 可选弱规则兜底：weak_json（brackets+json.loads+可选Schema），weak_regex（任意匹配）
- 仍输出窄表 6 列：id, metric, score, stratum, cluster, time_idx（兼容下游 TOST）
- 剔除 warmup；强制 json/general 双层分层；可写摘要与诊断

依赖：pandas；可选：jsonschema（仅当 --schema 时）
"""

import os, re, json, argparse, pandas as pd

# ---------- I/O ----------
def load_jsonl(p):
    rows = []
    with open(p, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line: 
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                # 跳过坏行
                pass
    return rows

def load_gold(p):
    if (not p) or (not os.path.exists(p)):
        return {}
    gold = {}
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): 
                continue
            o = json.loads(line)
            gid = int(o["id"])
            gold[gid] = {
                "must_include": [s.lower() for s in o.get("must_include", [])],
                "regex_any": o.get("regex_any", []),
                "stratum": o.get("stratum")
            }
    return gold

# ---------- Gold 规则打分 ----------
def judge_text(text, rule):
    t = (text or "").lower()
    for s in rule.get("must_include", []):
        if s in t:
            return 1.0
    for pat in rule.get("regex_any", []):
        try:
            if re.search(pat, t, flags=re.I | re.S):
                return 1.0
        except re.error:
            # 忽略坏正则
            continue
    return 0.0

# ---------- 弱规则（可选）----------
def bracket_like(t: str) -> bool:
    t = (t or "").strip()
    return (t.startswith("{") and t.endswith("}")) or (t.startswith("[") and t.endswith("]"))

def try_json_load(t: str):
    try:
        obj = json.loads((t or "").strip())
        return True, obj, ""
    except Exception as e:
        return False, None, str(e)[:160]

try:
    from jsonschema import Draft7Validator
except Exception:
    Draft7Validator = None

def make_validator(schema_path: str):
    if not schema_path:
        return None, None
    if Draft7Validator is None:
        raise RuntimeError("需要 jsonschema 依赖：pip install jsonschema")
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)
    version = schema.get("$id") or schema.get("version") or os.path.basename(schema_path)
    return Draft7Validator(schema), str(version)

def weak_json_score(text: str, validator) -> float:
    # 结构→解析→Schema（若有）；全通过给 1.0，否则 0.0
    if not bracket_like(text):
        return 0.0
    ok, obj, _ = try_json_load(text)
    if not ok:
        return 0.0
    if validator is not None:
        try:
            validator.validate(obj)
        except Exception:
            return 0.0
    return 1.0

def weak_regex_score(text: str, weak_pats) -> float:
    if not weak_pats:
        return 0.0
    t = (text or "")
    for pat in weak_pats:
        try:
            if re.search(pat, t, flags=re.I | re.S):
                return 1.0
        except re.error:
            continue
    return 0.0

# ---------- 分层 ----------
def is_json_task(o):
    if o.get("need_json") is True:
        return True
    p = (o.get("prompt", "") or "").lower()
    return ("json" in p) or ("{" in p) or ("[" in p)

# ---------- 主流程 ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help=r"runs\*.jsonl（单文件）")
    ap.add_argument("--gold", default=None, help="gold.jsonl（可选；含 must_include/regex_any）")
    ap.add_argument("--out", required=True, help="scores_*.csv")
    ap.add_argument("--metric", default="acc", help="质量主指标名（如 acc/factscore/attribution）")
    ap.add_argument("--cluster-col", default=None, help="使用日志中的列做 cluster（默认 session）")

    # —— 关键：无 gold 的样本如何处理 —— 
    ap.add_argument("--missing-policy", choices=["exclude","zero","weak_json","weak_regex"],
                    default="exclude",
                    help="exclude=从主指标剔除（推荐）；zero=记0（保守）；weak_json=JSON弱规则兜底；weak_regex=正则兜底")
    ap.add_argument("--schema", default=None, help="当 --missing-policy=weak_json 时可提供 AnswerContract Schema")
    ap.add_argument("--weak-regex", action="append", default=[],
                    help="当 --missing-policy=weak_regex 时可提供多个正则（可重复此参数）")

    # 审计/诊断
    ap.add_argument("--summary-json", default=None, help="写出摘要/指纹到此 JSON（可选）")
    ap.add_argument("--diag", default=None, help="写出诊断明细 CSV（可选）")

    args = ap.parse_args()

    logs = load_jsonl(args.log)
    gold = load_gold(args.gold)

    validator, schema_version = (None, None)
    if args.missing_policy == "weak_json" and args.schema:
        validator, schema_version = make_validator(args.schema)

    rows, diag = [], []
    n_total = 0
    n_warm = 0
    n_has_gold = 0
    n_missing_gold = 0
    n_scored = 0

    for o in logs:
        n_total += 1
        if o.get("in_warmup", False):
            n_warm += 1
            continue

        i = int(o.get("id", 0) or 0)
        prompt = o.get("prompt", "")
        output = o.get("output") or o.get("text") or ""

        need_json = is_json_task(o)
        stratum = "json" if need_json else "general"

        # gold 规则
        rule = gold.get(i, None)
        if rule is not None:
            n_has_gold += 1
            sc = judge_text(output, rule)
            policy_used = "gold"
            weak_detail = ""
        else:
            n_missing_gold += 1
            # 应用“无 gold”策略
            if args.missing_policy == "exclude":
                # 不产出行，保持主口径干净；双方臂使用相同 gold 时会对齐样本集
                policy_used = "exclude"
                weak_detail = ""
                if args.diag:
                    diag.append({
                        "id": i, "need_json": need_json, "policy": policy_used,
                        "score": None, "reason": "no_gold_excluded"
                    })
                continue
            elif args.missing_policy == "zero":
                sc = 0.0
                policy_used = "zero"
                weak_detail = "no_gold->0"
            elif args.missing_policy == "weak_json":
                sc = weak_json_score(output, validator)
                policy_used = "weak_json"
                weak_detail = f"schema={bool(validator)}"
            elif args.missing_policy == "weak_regex":
                sc = weak_regex_score(output, args.weak_regex)
                policy_used = "weak_regex"
                weak_detail = f"n_pats={len(args.weak_regex)}"
            else:
                # 理论不达
                sc = 0.0
                policy_used = "unknown"
                weak_detail = ""

        cluster = o.get(args.cluster_col) if args.cluster_col else (o.get("mode", "session"))
        time_idx = o.get("time_idx", o.get("id", 0))

        rows.append(dict(
            id=i, metric=args.metric, score=float(sc),
            stratum=stratum, cluster=str(cluster), time_idx=int(time_idx)
        ))
        n_scored += 1

        if args.diag:
            diag.append({
                "id": i,
                "need_json": bool(need_json),
                "policy": policy_used,
                "weak_detail": weak_detail,
                "score": float(sc),
                "final_path": o.get("final_path"),
                "prompt_len": int(o.get("prompt_len", 0) or 0),
                "plan_complexity": o.get("plan_complexity"),
            })

    # 输出窄表（保持与下游一致）
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("打分结果为空。请检查 --missing-policy、gold 文件与日志内容。")
    # 兜底：任何非 "json" 的都并为 "general"
    df["stratum"] = df["stratum"].apply(lambda s: "json" if str(s).lower()=="json" else "general")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False, encoding="utf-8")
    print(f"[OK] 写出: {args.out}  样本={len(df)}  分层={df['stratum'].value_counts().to_dict()}")

    # 诊断可选
    if args.diag:
        ddf = pd.DataFrame(diag)
        ddf.to_csv(args.diag, index=False, encoding="utf-8")
        print(f"[OK] 诊断: {args.diag}  行数={len(ddf)}")

    # 摘要/指纹
    if args.summary_json:
        summary = {
            "n_total_seen": int(n_total),
            "n_warm_skipped": int(n_warm),
            "n_scored": int(n_scored),
            "n_has_gold": int(n_has_gold),
            "n_missing_gold": int(n_missing_gold),
            "missing_policy": args.missing_policy,
            "schema_version": schema_version,
            "weak_regex_count": len(args.weak_regex) if args.missing_policy=="weak_regex" else 0
        }
        with open(args.summary_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"[OK] 摘要: {args.summary_json}")

if __name__ == "__main__":
    main()
