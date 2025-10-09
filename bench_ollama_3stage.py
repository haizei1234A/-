# -*- coding: utf-8 -*-
"""
"""

from __future__ import annotations

import os
import json
import time
import math
import uuid
import glob
import random
import argparse
import datetime
import platform
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Callable

import numpy as np
import pandas as pd
import requests

# ---------- 可选依赖 ----------
try:
    from jsonschema import Draft7Validator
except Exception:
    Draft7Validator = None

try:
    from tdigest import TDigest
    import tdigest as _td_mod
    TDIGEST_VERSION = getattr(_td_mod, "__version__", "unknown")
except Exception:
    TDigest = None
    TDIGEST_VERSION = "unknown"

try:
    import tiktoken  # 更准确的 token 计数（可选）
except Exception:
    tiktoken = None

# ---------- 常量 ----------
EPS = 1e-12
MIN_JSON_LEN = 40
DEFAULT_P95_DELTA = 0.10
DEFAULT_JSON_FAIL_MAX = 0.005

# 事件指纹默认
BILLING_VERSION = "vNext-5part"
TDIGEST_DEFAULT_COMPRESSION = 200

# ---------- 环境 ----------
OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_KEEPALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "5m")
LIGHT_MODEL = os.getenv("OLLAMA_LIGHT", "qwen3:0.6b")
STD_MODEL   = os.getenv("OLLAMA_STD",   "qwen3:1.7b")
ENH_MODEL   = os.getenv("OLLAMA_ENH",   "qwen3:4b")


# ---------- 工具函数：分位数 ----------
def _hd(values: Iterable[float], q: float) -> float:
    """Harrell–Davis 分位数点估计（小样本更稳健）。"""
    x = sorted([float(v) for v in values if not pd.isna(v)])
    n = len(x)
    if n == 0:
        return np.nan
    if n == 1:
        return float(x[0])
    q = max(EPS, min(1.0 - EPS, float(q)))
    log_q, log_1q = math.log(q), math.log(1 - q)
    ws = []
    for i in range(1, n + 1):
        a, b = i, n - i + 1
        logB = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
        ws.append(math.exp((i - 1) * log_q + (n - i) * log_1q - logB))
    s = sum(ws)
    ws = [w / s for w in ws]
    return float(sum(w * xi for w, xi in zip(ws, x)))


def q50(values: Iterable[float]) -> float:
    return _hd(values, 0.5)


def q95(values: Iterable[float]) -> float:
    return _hd(values, 0.95)


def tdigest_quantile(values: Iterable[float], q: float, compression: int = TDIGEST_DEFAULT_COMPRESSION) -> float:
    """在线/可并合分位（无 tdigest 时回退 HD）。"""
    vals = [float(v) for v in values if not pd.isna(v)]
    if not vals:
        return np.nan
    if TDigest is None:
        return _hd(vals, q)
    d = TDigest(compression=compression)
    for v in vals:
        d.update(v)
    return float(d.quantile(q))


# ---------- CO-corrected（periodic fill） ----------
def co_correct_periodic_fill(lat_ms_list: Iterable[float], period_ms: int, max_fills: int = 120) -> List[float]:
    """
    对每个样本 L，补入 L-P, L-2P, ... > 0 的“应采未采”样本。
    max_fills：每个 L 的最大补样次数（工程护栏，避免极端长尾时的爆炸）
    """
    if not period_ms or period_ms <= 0:
        return [float(x) for x in lat_ms_list if not pd.isna(x)]
    out: List[float] = []
    for L in lat_ms_list:
        if pd.isna(L):
            continue
        L = float(L)
        out.append(L)
        if L <= period_ms:
            continue
        # 允许的最大补样次数
        kmax = min(int((L - EPS) // period_ms), max_fills)
        # 生成 L - j*P
        for j in range(1, kmax + 1):
            over = L - j * period_ms
            if over > 0:
                out.append(over)
            else:
                break
    return out


# ---------- 生存分析：KM 分位（右删失） ----------
def km_quantile(times_ms: Iterable[float], observed: Iterable[bool], q: float = 0.95) -> float:
    """
    Kaplan–Meier 估计的分位数（若不可达，返回 NaN）。
    times_ms: 事件或删失时间；observed: True=事件，False=右删失。
    """
    arr = [(float(t), bool(o)) for t, o in zip(times_ms, observed) if not pd.isna(t)]
    if not arr:
        return np.nan
    arr.sort(key=lambda x: x[0])
    uniq = sorted(set(t for t, _ in arr))
    n_total = len(arr)
    s = 1.0  # 生存函数 S(t)
    idx = 0
    for t in uniq:
        # 风险集大小 n_i（进入该时刻前尚未失败/删失）
        while idx < n_total and arr[idx][0] < t:
            idx += 1
        n_i = n_total - idx
        d_i = sum(1 for tt, ob in arr if tt == t and ob)
        if n_i <= 0:
            continue
        s *= (1.0 - d_i / n_i)
        if 1.0 - s >= q:
            return float(t)
    return np.nan


# ---------- Winsor（敏感性） ----------
def winsorize(values: Iterable[float], p: float = 0.999) -> List[float]:
    xs = [float(v) for v in values if not pd.isna(v)]
    if not xs:
        return xs
    lo = _hd(xs, 1 - p)
    hi = _hd(xs, p)
    return [min(max(v, lo), hi) for v in xs]


# ---------- Bootstrap（BCa/Percentile） ----------
def _percentile_ci(samples: np.ndarray, ci: float) -> Tuple[float, float]:
    alpha = (1 - ci) / 2
    lo = float(np.quantile(samples, alpha))
    hi = float(np.quantile(samples, 1 - alpha))
    return lo, hi


def _bca_ci(samples: np.ndarray, theta_hat: float, jack: np.ndarray, ci: float) -> Tuple[float, float]:
    # 偏倚修正 z0
    prop = np.mean(samples < theta_hat) + EPS
    from math import erfinv, sqrt
    z0 = math.sqrt(2) * erfinv(2 * prop - 1)

    # 加速因子 a（jackknife）
    jmean = float(np.mean(jack))
    num = np.sum((jmean - jack) ** 3)
    den = 6.0 * (np.sum((jmean - jack) ** 2) ** 1.5 + EPS)
    a = float(num / (den + EPS))

    def z(p: float) -> float:
        from math import erfinv, sqrt
        return math.sqrt(2) * erfinv(2 * p - 1)

    alpha = (1 - ci) / 2
    zlo, zhi = z(alpha), z(1 - alpha)

    def pct(adj_z: float) -> float:
        from math import erf, sqrt
        p = 0.5 * (1 + erf(((z0 + adj_z) / (1 - a * (z0 + adj_z))) / math.sqrt(2)))
        p = max(0.0, min(1.0, p))
        return float(np.quantile(samples, p))

    return pct(zlo), pct(zhi)


def bootstrap_stat_ci(
    sample: Iterable[float],
    stat_fn: Callable[[np.ndarray], float],
    B: int = 1000,
    ci: float = 0.95,
    seed: Optional[int] = None,
    method: str = "bca",
) -> Tuple[float, float, float]:
    """
    通用一维自助：返回 (theta_hat, lo, hi)。
    """
    x = np.array([v for v in sample if not pd.isna(v)], dtype=float)
    n = x.size
    if n == 0:
        return np.nan, np.nan, np.nan

    rng = np.random.default_rng(seed)
    theta_hat = float(stat_fn(x))

    thetas = np.empty(B, dtype=float)
    for b in range(B):
        idx = rng.integers(0, n, n)
        thetas[b] = float(stat_fn(x[idx]))

    if method == "percentile":
        lo, hi = _percentile_ci(thetas, ci)
        return theta_hat, lo, hi

    # —— BCa ——（jackknife）
    jack = np.empty(n, dtype=float)
    for i in range(n):
        jack[i] = float(stat_fn(np.delete(x, i)))
    lo, hi = _bca_ci(thetas, theta_hat, jack, ci)
    return theta_hat, lo, hi


# ---------- FDR：Benjamini–Hochberg（BH-95） ----------
def bh_adjust(pvals: List[Optional[float]], q: float = 0.10) -> Tuple[List[float], List[bool]]:
    p = np.array([pv if (pv is not None and not np.isnan(pv)) else 1.0 for pv in pvals], dtype=float)
    n = p.size
    order = np.argsort(p)
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(1, n + 1)
    qvals = p * n / ranks
    qvals_sorted = np.minimum.accumulate(qvals[order][::-1])[::-1]
    q_final = np.empty(n, dtype=float)
    q_final[order] = np.minimum(qvals_sorted, 1.0)
    reject = q_final <= q
    return q_final.tolist(), reject.tolist()


# ---------- Token 计数 ----------
def count_tokens(text: str, mode: str = "auto") -> int:
    if not text:
        return 0
    if mode == "auto" and tiktoken is not None:
        try:
            enc = tiktoken.get_encoding("cl100k_base")
        except Exception:
            enc = tiktoken.encoding_for_model("gpt-4")
        return int(len(enc.encode(text)))
    # 近似：1 token ≈ 4 chars（中英文混排可接受的工程近似）
    return max(1, int(len(text) / 4))


# ---------- Stratification ----------
def ctx_bucket_by_len(L: int) -> str:
    if L <= 120:
        return "short"
    if L <= 400:
        return "medium"
    return "long"


def stratify_row(event_row: Dict[str, Any], region: str, az: str) -> Dict[str, str]:
    prompt = event_row.get("prompt", "")
    return {
        "task": event_row.get("task", "general"),
        "ctx_bucket": ctx_bucket_by_len(len(prompt)),
        "tool_use": str(bool(event_row.get("tool_use", False))),
        "provider_model": f"ollama/{event_row.get('final_path','')}",
        "region": region or "",
        "az": az or "",
        "cold_warm": "cold" if event_row.get("in_warmup") else "warm",
        "cache_hit_prompt": str(False),
        "cache_hit_kv": str(False),
    }


# ---------- Prompt 流式读取 ----------
class EmptyPromptFileError(RuntimeError):
    """Raised when a prompt file yields no usable prompts after a full pass."""


class PromptCycler:
    """逐行读取 JSONL（字典行优先取 'prompt' 字段），读到 EOF 自动回绕。"""
    def __init__(self, path: str) -> None:
        self.path = path
        if not os.path.exists(path):
            raise FileNotFoundError(path)

    def take(self, n: int) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        rewound_without_prompts = False
        with open(self.path, "r", encoding="utf-8-sig") as f:
            added_in_pass = 0
            while len(out) < n:
                line = f.readline()
                if not line:
                    if out:
                        f.seek(0)
                        continue
                    if rewound_without_prompts:
                        raise EmptyPromptFileError(
                            f"{self.path} yielded no usable prompts after a full pass"
                        )
                    f.seek(0)
                    rewound_without_prompts = True
                    continue
                if line.startswith("#") or line.startswith("//"):
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        out.append(obj)
                    else:
                        out.append({"prompt": str(obj)})
                    added_in_pass += 1
                except Exception:
                    out.append({"prompt": line})
                rewound_without_prompts = False
        return out


# ---------- Ollama Client ----------
class OllamaClient:
    def __init__(self, base: str, keep_alive: str) -> None:
        self.base = base.rstrip("/")
        self.keep_alive = keep_alive
        self.session = requests.Session()

    def generate(
        self,
        model: str,
        prompt: str,
        num_ctx: int,
        num_predict: int,
        timeout_s: int,
        temperature: float,
        top_p: float,
        need_json: bool = False,
    ) -> Dict[str, Any]:
        url = f"{self.base}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {
                "num_ctx": int(num_ctx),
                "num_predict": int(num_predict),
                "temperature": float(temperature),
                "top_p": float(top_p),
            },
        }
        if need_json:
            payload["format"] = "json"

        t0 = time.perf_counter()
        try:
            r = self.session.post(url, json=payload, timeout=timeout_s)
            wall_ms = (time.perf_counter() - t0) * 1000.0
            r.raise_for_status()
            data = r.json()
            total_ms = (data.get("total_duration", 0) / 1e6) or wall_ms
            return {
                "text": data.get("response", "") or "",
                "e2e_ms": float(total_ms),
                "eval_tokens": int(data.get("eval_count", 0)),
                "right_censored": False,
                "error": "",
                "error_kind": "",
            }
        except requests.exceptions.Timeout as e:
            wall_ms = (time.perf_counter() - t0) * 1000.0
            return {
                "text": "",
                "e2e_ms": float(wall_ms),
                "eval_tokens": 0,
                "right_censored": True,   # 右删失：超时
                "error": f"timeout:{str(e)}",
                "error_kind": "timeout",
            }
        except requests.exceptions.RequestException as e:
            wall_ms = (time.perf_counter() - t0) * 1000.0
            return {
                "text": "",
                "e2e_ms": float(wall_ms),
                "eval_tokens": 0,
                "right_censored": False,  # 真实错误
                "error": f"request_error:{str(e)}",
                "error_kind": "request",
            }
        except Exception as e:
            wall_ms = (time.perf_counter() - t0) * 1000.0
            return {
                "text": "",
                "e2e_ms": float(wall_ms),
                "eval_tokens": 0,
                "right_censored": False,
                "error": f"error:{str(e)}",
                "error_kind": "unknown",
            }


# ---------- Planner/QAR/受控执行 ----------
KEYWORDS_HEAVY = ["法律","合规","财务","医学","诊断","复杂算法","伪代码","正则","多步骤","RAG","检索","对比","排序","解释原因"]
CODE_HINTS = ["代码", "regex", "正则", "python", "js", "java", "sql", "shell", "示例代码", "伪代码"]

def is_json_task(prompt: str) -> bool:
    p = (prompt or "").lower()
    return ("json" in p) or ("{" in p) or ("[" in p)

def looks_like_code(prompt: str) -> bool:
    p = (prompt or "").lower()
    if any(k in p for k in ["regex","python","java","sql","shell","code","snippet"]):
        return True
    return any(k in (prompt or "") for k in CODE_HINTS)

def plan(prompt: str) -> Dict[str, Any]:
    L = len(prompt or "")
    marks = sum((prompt or "").count(x) for x in ["?","？",":","：","\n"])
    heavy = any(k.lower() in (prompt or "").lower() for k in KEYWORDS_HEAVY)
    score = (L/200.0) + (marks/6.0) + (0.8 if heavy else 0.0)  # ~[0,1.6]
    return {"len": L, "marks": marks, "heavy_kw": heavy, "complexity": max(0.0, min(1.6, score))}

def route(score: float, thr_std: float, thr_enh: float) -> Tuple[str, str]:
    if score < float(thr_std):
        return "light", f"complexity={score:.2f} < {thr_std}"
    if score < float(thr_enh):
        return "std",   f"complexity={score:.2f} < {thr_enh}"
    return "enh",      f"complexity={score:.2f} >= {thr_enh}"

def json_contract_ok(text: str, schema_validator: Optional[Draft7Validator]) -> Tuple[bool, str]:
    try:
        obj = json.loads((text or "").strip())
    except Exception:
        return False, "not_json"
    if schema_validator is None:
        return True, ""
    try:
        schema_validator.validate(obj)
        return True, ""
    except Exception as e:
        return False, f"schema:{str(e)[:200]}"

def needs_upgrade(prompt: str, text: str, need_json: bool) -> str:
    t = (text or "").strip()
    if need_json:
        if not (t.startswith("{") or t.startswith("[")) or not (t.endswith("}") or t.endswith("]")):
            return "expect_json_but_not_json"
    if len(t) < MIN_JSON_LEN:
        return "too_short"
    return ""

_REFUSAL_HINTS = ["抱歉", "无法帮助", "不便提供", "sorry", "cannot help", "cannot comply"]
def is_refusal(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in _REFUSAL_HINTS)

def sleep_to_period(t_start: float, period_ms: int) -> None:
    if period_ms and period_ms > 0:
        elap = (time.perf_counter() - t_start) * 1000.0
        rest = period_ms - elap
        if rest > 0:
            time.sleep(rest / 1000.0)

class Gateway:
    def __init__(self, client: OllamaClient, args: argparse.Namespace) -> None:
        self.client = client
        self.args = args
        self.model_map = {"light": LIGHT_MODEL, "std": STD_MODEL, "enh": ENH_MODEL}

    def controlled_execute(self, first_path: str, prompt: str) -> Dict[str, Any]:
        order = ["light","std","enh"]
        need_json = is_json_task(prompt)

        # 楼层兜底
        floor = self.args.def_floor
        if need_json:
            floor = self.args.json_floor
        elif looks_like_code(prompt):
            floor = self.args.code_floor
        if order.index(first_path) < order.index(floor):
            first_path = floor

        # JSON 任务上调 cap & 严格提示
        cap = max(self.args.cap_enh, 192) if need_json else self.args.cap_enh
        prompt_eff = ("只输出 JSON，不要解释与多余文本。\n" + (prompt or "")) if need_json else (prompt or "")

        cur = first_path
        attempts = 0
        upgrades = 0
        repaired = False
        last_out: Dict[str, Any] = {
            "text": "", "e2e_ms": np.nan, "eval_tokens": 0,
            "right_censored": False, "error": "", "error_kind": ""
        }
        reason = ""

        while attempts < (self.args.max_upgrades + 1):
            last_out = self.client.generate(
                self.model_map[cur], prompt_eff, self.args.num_ctx, cap,
                self.args.timeout, self.args.temperature, self.args.top_p, need_json=need_json
            )
            attempts += 1
            if last_out.get("right_censored", False):  # 超时 → 右删失；不做同层修复
                break
            reason = needs_upgrade(prompt_eff, last_out.get("text",""), need_json)
            if (not reason) or cur == "enh":
                break

            # 同层修复一次
            if attempts < (self.args.max_upgrades + 1):
                out2 = self.client.generate(
                    self.model_map[cur], prompt_eff, self.args.num_ctx, cap,
                    self.args.timeout, self.args.temperature, self.args.top_p, need_json=need_json
                )
                attempts += 1
                r2 = needs_upgrade(prompt_eff, out2.get("text",""), need_json)
                if not r2:
                    last_out = out2
                    reason = "fixed_once:" + reason
                    repaired = True
                    break

            # 受控升级
            i = order.index(cur)
            if i < len(order) - 1 and upgrades < self.args.max_upgrades:
                upgrades += 1
                cur = order[i + 1]
            else:
                break

        return {
            "final_path": cur,
            "attempts": attempts,
            "upgrades": upgrades,
            "repaired": repaired,
            "upgrade_reason": reason,
            **last_out
        }


# ---------- 质量打分（可选） ----------
def quality_score(event_row: Dict[str, Any], mode: str = "auto") -> float:
    """基于 prompts.jsonl 的 label/answers 字段进行粗评分（无则 NaN）。"""
    if mode == "none":
        return np.nan
    obj = event_row.get("_prompt_obj")
    if not isinstance(obj, dict):
        return np.nan
    gold = obj.get("label") or obj.get("answer") or None
    answers = obj.get("answers")
    out = (event_row.get("output") or "").strip().lower()
    if gold:
        g = str(gold).strip().lower()
        if mode in ("auto","exact"):
            return 1.0 if out == g else 0.0
        if mode == "substring":
            return 1.0 if g in out else 0.0
    if answers and isinstance(answers, list) and answers:
        al = [str(a).strip().lower() for a in answers]
        if mode in ("auto","exact"):
            return 1.0 if out in al else 0.0
        if mode == "substring":
            return 1.0 if any(a in out for a in al) else 0.0
    return np.nan


# ---------- 统计聚合 ----------
@dataclass
class StatsResult:
    summary: Dict[str, Any]
    warm_df: pd.DataFrame
    df_all: pd.DataFrame


def compute_stats_and_summary(
    rows: List[Dict[str, Any]],
    args: argparse.Namespace,
    tdigest_version: str = TDIGEST_VERSION
) -> StatsResult:
    df = pd.DataFrame(rows)
    warm_df = df[~df["in_warmup"]].copy()

    # --- P50/P95 Raw（overall & warm） ---
    overall_lat = df["e2e_ms"].tolist()
    warm_lat = warm_df["e2e_ms"].tolist()
    overall_p50 = q50(overall_lat)
    overall_p95 = q95(overall_lat)
    warm_p50 = q50(warm_lat)
    warm_p95 = q95(warm_lat)

    # --- CO-corrected ---
    if args.co_correction == "periodic_fill":
        overall_lat_co = co_correct_periodic_fill(overall_lat, args.period_ms, args.max_co_fills)
        warm_lat_co = co_correct_periodic_fill(warm_lat, args.period_ms, args.max_co_fills)
    else:
        overall_lat_co = overall_lat
        warm_lat_co = warm_lat
    overall_p95_co = tdigest_quantile(overall_lat_co, 0.95, compression=args.tdigest_compression)
    warm_p95_co = tdigest_quantile(warm_lat_co, 0.95, compression=args.tdigest_compression)

    # --- 删失：KM 与 SLO-cap（warm-only） ---
    warm_obs = ~(warm_df["right_censored"].astype(bool))
    warm_obs_list = warm_obs.tolist()
    warm_lat_list = warm_df["e2e_ms"].tolist()
    warm_p95_km = km_quantile(warm_lat_list, warm_obs_list, q=0.95)
    timeout_cap_ms = args.timeout_cap_ms if args.timeout_cap_ms else (args.timeout * 1000)
    warm_lat_cap = [(timeout_cap_ms if (not ob) else t) for t, ob in zip(warm_lat_list, warm_obs_list)]
    warm_p95_cap = q95(warm_lat_cap)
    timeouts_rate = float((~warm_obs).mean() * 100.0)
    censoring_mode = "KM / SLO-cap"

    # 路由分布与 r_costly
    dist = warm_df["final_path"].value_counts(normalize=True) * 100.0
    light_pct = float(dist.get("light", 0.0))
    std_pct = float(dist.get("std", 0.0))
    enh_pct = float(dist.get("enh", 0.0))
    r_costly_pct = float((warm_df["final_path"] == "enh").mean() * 100.0)

    # JSON 失败率（仅针对需要 JSON 的样本）
    need_json_mask = warm_df["need_json"] == True
    if need_json_mask.any() and "json_ok" in warm_df:
        json_fail_rate = float((~warm_df.loc[need_json_mask, "json_ok"]).mean() * 100.0)
    else:
        json_fail_rate = 0.0

    # ITT 衍生率
    retry_rate_pct = float((warm_df["retry_count"] > 0).mean() * 100.0)
    refusal_rate_pct = float((warm_df["refusal"] == True).mean() * 100.0)

    # 分层标签与权重
    strat_rows = []
    for _, r in warm_df.iterrows():
        s = stratify_row(r, args.region, args.az)
        s["stratum"] = "|".join([s["task"], s["ctx_bucket"], s["tool_use"], s["provider_model"], s["region"],
                                 s["az"], s["cold_warm"], s["cache_hit_prompt"], s["cache_hit_kv"]])
        strat_rows.append(s)
    strat_df = pd.DataFrame(strat_rows)
    warm_df = pd.concat([warm_df.reset_index(drop=True), strat_df.reset_index(drop=True)], axis=1)
    warm_df["weight"] = 1.0
    if getattr(args, "poststrat_weights", None):
        try:
            wdf = pd.read_csv(args.poststrat_weights)
            weight_map = {str(r["stratum"]): float(r["weight"]) for _, r in wdf.iterrows()}
            warm_df["weight"] = warm_df["stratum"].map(weight_map).fillna(1.0)
        except Exception:
            pass

    # 质量分（可选）
    warm_df["q_primary"] = warm_df.apply(lambda r: quality_score(dict(r), mode=args.judge_mode), axis=1)

    summary = {
        # 基本元数据
        "mode": args.mode,
        "platform": platform.platform(),
        "light_model": LIGHT_MODEL, "std_model": STD_MODEL, "enh_model": ENH_MODEL,
        "N_total": int(len(df)), "N_eval": int(len(warm_df)),
        "num_ctx": int(args.num_ctx), "cap_enh": int(args.cap_enh),
        "temperature": float(args.temperature), "top_p": float(args.top_p),

        # 四口径 P95
        "overall_p50_ms": float(overall_p50),     "overall_p95_ms": float(overall_p95),
        "overall_p95_ms_co": float(overall_p95_co),
        "warm_p50_ms": float(warm_p50),           "warm_p95_ms": float(warm_p95),
        "warm_p95_ms_co": float(warm_p95_co),

        # 删失与口径披露（warm）
        "warm_p95_ms_km": float(warm_p95_km),
        "warm_p95_ms_cap": float(warm_p95_cap),
        "timeouts_rate_pct": float(timeouts_rate),
        "censoring_mode": censoring_mode,

        # 工作量（E2E 口径近似）
        "avg_eval_ms": float(warm_df["e2e_ms"].mean()),
        "sum_eval_ms": float(warm_df["e2e_ms"].sum()),

        # 路由/成本侧
        "light_pct": float(light_pct), "std_pct": float(std_pct), "enh_pct": float(enh_pct),
        "renh_pct": float(enh_pct), "r_costly_pct": float(r_costly_pct),

        # 合规与 ITT
        "json_fail_rate_pct": float(json_fail_rate),
        "retry_rate_pct": float(retry_rate_pct),
        "refusal_rate_pct": float(refusal_rate_pct),

        # 审计指纹
        "sla": args.sla,
        "period_ms": int(args.period_ms),
        "sampler_period_ms": int(args.period_ms),
        "co_correction": str(args.co_correction),
        "tdigest_compression": int(args.tdigest_compression),
        "tdigest_version": tdigest_version or "unknown",
        "warmup_window_ms": int(args.warmup_window_ms),
        "periodic_load": True,
        "hdr_sigfig": None,
        "seed": args.seed if args.seed is not None else None,
        "engine": "ollama",
        "quant_config": None,
        "model_build": None, "router_build": None, "planner_build": None,

        # 路由/楼层参数
        "thr_std": float(args.thr_std), "thr_enh": float(args.thr_enh),
        "json_floor": str(args.json_floor), "code_floor": str(args.code_floor), "def_floor": str(args.def_floor),
        "max_upgrades": int(args.max_upgrades),
        "force_light_len": int(args.force_light_len), "force_light_score": float(args.force_light_score),
    }

    return StatsResult(summary=summary, warm_df=warm_df, df_all=df)


# ---------- 家族检验（A/B/C） ----------
def family_C_json_guard(warm_df: pd.DataFrame, q: float) -> List[Dict[str, Any]]:
    g = warm_df.copy()
    g["json_need_and_fail"] = (g["need_json"] == True) & (g["json_ok"] == False)
    tests: List[Dict[str, Any]] = []
    for s, sdf in g.groupby("stratum"):
        n_need = int((sdf["need_json"] == True).sum())
        x_fail = int((sdf["json_need_and_fail"]).sum())
        if n_need == 0:
            pval = None
        else:
            # 单侧：H0 p>=p0；H1 p<p0（p0=0.5%）
            p0 = DEFAULT_JSON_FAIL_MAX
            phat = x_fail / max(1, n_need)
            sd = math.sqrt(p0 * (1 - p0) / max(1, n_need))
            z = (phat - p0) / (sd + EPS)
            from math import erf, sqrt
            pval = 0.5 * (1 + erf(z / math.sqrt(2)))
        tests.append({"family": "C", "stratum": s, "n_need": n_need, "x_fail": x_fail, "raw_p": pval})
    qvals, reject = bh_adjust([t["raw_p"] for t in tests], q=q)
    for t, qv, rj in zip(tests, qvals, reject):
        t["q_value"] = qv; t["reject"] = bool(rj)
    return tests


def _bootstrap_guardrail_p95(
    g_lat: List[float], b_lat: List[float], delta: float, B: int, compression: int, seed: Optional[int]
) -> float:
    """返回单侧 p 值：Pr[ g95 <= (1+δ)*b95 ] 的补概率。"""
    if len(g_lat) == 0 or len(b_lat) == 0:
        return np.nan
    rng = np.random.default_rng(seed)
    cnt = 0
    ng, nb = len(g_lat), len(b_lat)
    for _ in range(B):
        gi = rng.integers(0, ng, ng)
        bi = rng.integers(0, nb, nb)
        g95_b = tdigest_quantile([g_lat[i] for i in gi], 0.95, compression=compression)
        b95_b = tdigest_quantile([b_lat[i] for i in bi], 0.95, compression=compression)
        if g95_b <= (1.0 + delta) * b95_b:
            cnt += 1
    pval = 1.0 - (cnt / max(1, B))  # 右侧拒绝域
    return float(pval)


def _family_B_run_one(
    item: Tuple[int, str, List[float], List[float]],
    p95_delta: float,
    bootstrap_B: int,
    tdigest_compression: int,
    seed: Optional[int],
) -> Tuple[int, str, float, float, float]:
    idx, s, g_lat, b_lat = item
    g95s = tdigest_quantile(g_lat, 0.95, compression=tdigest_compression)
    b95s = tdigest_quantile(b_lat, 0.95, compression=tdigest_compression)
    pval = _bootstrap_guardrail_p95(
        g_lat,
        b_lat,
        p95_delta,
        bootstrap_B,
        tdigest_compression,
        seed,
    )
    return idx, s, g95s, b95s, pval


def family_B_p95_guardrail(
    warm_df: pd.DataFrame,
    base_warm_df: Optional[pd.DataFrame],
    args: argparse.Namespace
) -> List[Dict[str, Any]]:
    tests: List[Dict[str, Any]] = []
    if base_warm_df is None or base_warm_df.empty:
        return tests

    # 基线同分层构造
    base_warm = base_warm_df.copy()
    base_warm["stratum"] = base_warm.apply(
        lambda r: "|".join([
            stratify_row(r, args.region, args.az)["task"],
            stratify_row(r, args.region, args.az)["ctx_bucket"],
            stratify_row(r, args.region, args.az)["tool_use"],
            stratify_row(r, args.region, args.az)["provider_model"],
            args.region, args.az,
            "cold" if r.get("in_warmup") else "warm", "False", "False"
        ]),
        axis=1
    )

    # 并行/串行
    work_items: List[Tuple[int, str, List[float], List[float]]] = []
    for idx, (s, sdf) in enumerate(warm_df.groupby("stratum")):
        g_lat = sdf["e2e_ms"].tolist()
        b_lat = base_warm[base_warm["stratum"] == s]["e2e_ms"].tolist()
        if args.co_correction == "periodic_fill":
            g_lat = co_correct_periodic_fill(g_lat, args.period_ms, args.max_co_fills)
            b_lat = co_correct_periodic_fill(b_lat, args.period_ms, args.max_co_fills)
        work_items.append((idx, s, g_lat, b_lat))

    results: List[Tuple[int, str, float, float, float]] = []
    if args.workers and args.workers > 0:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=int(args.workers)) as ex:
            futs = [
                ex.submit(
                    _family_B_run_one,
                    it,
                    args.p95_delta,
                    args.bootstrap_B,
                    args.tdigest_compression,
                    args.seed,
                )
                for it in work_items
            ]
            for fu in as_completed(futs):
                results.append(fu.result())
    else:
        for it in work_items:
            results.append(
                _family_B_run_one(
                    it,
                    args.p95_delta,
                    args.bootstrap_B,
                    args.tdigest_compression,
                    args.seed,
                )
            )

    results.sort(key=lambda item: item[0])

    for _, s, g95s, b95s, pval in results:
        tests.append({"family": "B", "stratum": s, "p95_guard_delta": args.p95_delta, "g95": g95s, "b95": b95s, "raw_p": pval})

    qvals, reject = bh_adjust([t["raw_p"] for t in tests], q=args.fdr_q)
    for t, qv, rj in zip(tests, qvals, reject):
        t["q_value"] = qv; t["reject"] = bool(rj)
    return tests


def family_A_tost_placeholder(
    warm_df: pd.DataFrame, base_warm_df: Optional[pd.DataFrame], epsilon_q: float
) -> List[Dict[str, Any]]:
    """如无质量标注，输出结构位，待接入 judge。"""
    tests: List[Dict[str, Any]] = []
    if base_warm_df is None or base_warm_df.empty:
        return tests
    for s, sdf in warm_df.groupby("stratum"):
        if sdf["q_primary"].notna().any():
            # 这里保留结构位（需要基线同口径质量）；接入后可计算两单侧 p 值并做 FDR
            tests.append({"family": "A", "stratum": s, "epsilon_q": epsilon_q, "raw_p": None, "note": "placeholder; supply baseline quality to enable TOST"})
        else:
            tests.append({"family": "A", "stratum": s, "epsilon_q": epsilon_q, "raw_p": None, "note": "no quality labels"})
    return tests


# ---------- 落盘 ----------
def latest(path_glob: str) -> Optional[str]:
    cand = sorted(glob.glob(path_glob), key=os.path.getmtime, reverse=True)
    return cand[0] if cand else None

def write_outputs(
    df_all: pd.DataFrame, summary: Dict[str, Any],
    tests_A: List[Dict[str, Any]], tests_B: List[Dict[str, Any]], tests_C: List[Dict[str, Any]],
    args: argparse.Namespace, censoring_mode: str
) -> None:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"{args.mode}_{ts}"
    os.makedirs("runs", exist_ok=True)

    # 事件明细
    events_path = os.path.join("runs", f"{base}_events.jsonl")
    df_all.to_json(events_path, orient="records", lines=True, force_ascii=False)

    # 汇总 JSON
    summary_path = os.path.join("runs", f"{base}_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False, indent=2))

    # metrics_summary.parquet（无 pyarrow 时回退 CSV）
    metrics_df = pd.DataFrame([summary])
    try:
        metrics_df.to_parquet(os.path.join("runs", f"{base}_metrics_summary.parquet"), index=False)
    except Exception:
        metrics_df.to_csv(os.path.join("runs", f"{base}_metrics_summary.csv"), index=False)

    # 家族检验 CSV
    if tests_A:
        pd.DataFrame(tests_A).to_csv(os.path.join("runs", f"{base}_tests_family_A.csv"), index=False)
    if tests_B:
        pd.DataFrame(tests_B).to_csv(os.path.join("runs", f"{base}_tests_family_B.csv"), index=False)
    if tests_C:
        pd.DataFrame(tests_C).to_csv(os.path.join("runs", f"{base}_tests_family_C.csv"), index=False)

    # bootstrap_meta.json & seed_manifest.txt
    meta = {
        "B": int(args.bootstrap_B),
        "ci_method": args.bootstrap_ci,
        "seed": int(args.seed) if args.seed is not None else None,
        "resample_unit": "request",
        "block_len": None,  # 将来接入 MBB/Stationary
        "censoring_mode": censoring_mode,
        "co_params": {"method": args.co_correction, "period_ms": args.period_ms, "max_fills": args.max_co_fills},
        "estimator": {"offline": "HD+Bootstrap", "online": "t-digest",
                      "tdigest_compression": args.tdigest_compression, "tdigest_version": TDIGEST_VERSION},
        "FDR": {"method": "BH-95", "q": args.fdr_q},
        "stratification": ["task","ctx_bucket","tool_use","provider/model","region/az","cold_warm","cache_hit"],
        "weights_file": args.poststrat_weights
    }
    with open(os.path.join("runs", f"{base}_bootstrap_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    with open(os.path.join("runs", f"{base}_seed_manifest.txt"), "w", encoding="utf-8") as f:
        f.write(str(args.seed if args.seed is not None else "None") + "\n")


# ---------- CLI / 主流程 ----------
def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["baseline_enh","gateway_3stage"], required=True)

    # 安全预设
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=3)

    ap.add_argument("--prompts", default="prompts.jsonl")
    ap.add_argument("--cap_enh", type=int, default=192)
    ap.add_argument("--num_ctx", type=int, default=1536)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--sla", choices=["complete","json","none"], default="complete")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--period-ms", dest="period_ms", type=int, default=0)

    # 路由阈值 + 楼层 + 升级次数（兼容别名）
    ap.add_argument("--thr_std", "--thr-std", "--threshold-std", type=float, default=0.40, dest="thr_std")
    ap.add_argument("--thr_enh", "--thr-enh", "--threshold-enh", type=float, default=0.85, dest="thr_enh")
    ap.add_argument("--json_floor", "--json-floor", default="std", dest="json_floor")
    ap.add_argument("--code_floor", "--code-floor", default="std", dest="code_floor")
    ap.add_argument("--def_floor", "--def-floor", default="light", dest="def_floor")
    ap.add_argument("--max_upgrades", "--max-upgrades", type=int, default=2, dest="max_upgrades")

    # 轻臂“快车道”
    ap.add_argument("--force_light_len", type=int, default=120)
    ap.add_argument("--force_light_score", type=float, default=0.30)

    # 度量/指纹
    ap.add_argument("--co-correction", choices=["none","periodic_fill"], default="periodic_fill")
    ap.add_argument("--tdigest-compression", type=int, default=TDIGEST_DEFAULT_COMPRESSION, dest="tdigest_compression")
    ap.add_argument("--warmup-window-ms", type=int, default=None, dest="warmup_window_ms")
    ap.add_argument("--json-schema", default=None, dest="json_schema")
    ap.add_argument("--max-co-fills", type=int, default=120, dest="max_co_fills")

    # 统计协议 & FDR
    ap.add_argument("--bootstrap-B", type=int, default=1000, dest="bootstrap_B")
    ap.add_argument("--bootstrap-samples", type=int, dest="bootstrap_B")  # 别名
    ap.add_argument("--bootstrap-ci", choices=["bca","percentile"], default="bca")
    ap.add_argument("--fdr-q", type=float, default=0.10)
    ap.add_argument("--epsilon-q", type=float, default=0.01)
    ap.add_argument("--p95-delta", type=float, default=DEFAULT_P95_DELTA)
    ap.add_argument("--judge-mode", choices=["auto","exact","substring","none"], default="auto")
    ap.add_argument("--poststrat-weights", default=None)
    ap.add_argument("--workers", type=int, default=0, help="并行 worker 数（家族B Bootstrap）")

    # 审计/环境
    ap.add_argument("--region", default="local")
    ap.add_argument("--az", default="local-az")
    ap.add_argument("--user-tenant", default="bench")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--tokenizer", choices=["auto","approx"], default="auto")

    # 超时封顶（SLO-cap 口径）
    ap.add_argument("--timeout-cap-ms", type=int, default=None)

    return ap


def main() -> None:
    args = build_arg_parser().parse_args()

    # seed：默认随机化；评测/复现可手动设置
    if args.seed is None:
        args.seed = random.randint(0, 2**31 - 1)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # warmup 窗口
    if args.warmup_window_ms is None:
        args.warmup_window_ms = int(max(0, args.warmup) * max(0, args.period_ms))

    os.makedirs("runs", exist_ok=True)

    # prompts（流式）
    empty_prompt_exc: Optional[EmptyPromptFileError] = None
    try:
        prompts = PromptCycler(args.prompts).take(args.n)
    except EmptyPromptFileError as exc:
        empty_prompt_exc = exc
        prompts = []
    if not prompts:
        if empty_prompt_exc is not None:
            raise RuntimeError("prompts.jsonl 为空") from empty_prompt_exc

    # JSON Schema（可选）
    schema_validator = None
    if args.json_schema and Draft7Validator is not None:
        try:
            with open(args.json_schema, "r", encoding="utf-8") as f:
                schema_validator = Draft7Validator(json.load(f))
        except Exception:
            schema_validator = None  # 降级继续跑

    # 客户端与网关
    client = OllamaClient(OLLAMA_BASE, OLLAMA_KEEPALIVE)
    gw = Gateway(client, args)

    # 逐请求执行
    rows: List[Dict[str, Any]] = []
    warm_left = args.warmup
    session_id = str(uuid.uuid4())

    for idx, obj in enumerate(prompts, 1):
        prompt = obj.get("prompt") if isinstance(obj, dict) else str(obj)
        t_period = time.perf_counter()

        # 规划/路由
        if args.mode == "baseline_enh":
            plan_report = {"complexity": None}
            chosen = "enh"
            route_decision = "always_enh"
        else:
            plan_report = plan(prompt)
            chosen, route_decision = route(plan_report["complexity"], args.thr_std, args.thr_enh)
            if not is_json_task(prompt) and not looks_like_code(prompt):
                if args.force_light_len > 0 and len(prompt) <= args.force_light_len:
                    if (plan_report.get("complexity") or 0.0) <= args.force_light_score:
                        chosen = "light"
                        route_decision += " | fast_lane=light"

        # 执行
        ret = gw.controlled_execute(chosen, prompt)
        ok = (ret.get("error", "") == "")

        # 固定节拍以匹配 CO 审计
        sleep_to_period(t_period, args.period_ms)

        # JSON 合规
        need_json = is_json_task(prompt)
        json_ok, json_violation = (True, "")
        if need_json:
            json_ok, json_violation = json_contract_ok(ret.get("text", ""), schema_validator)

        # 事件行（最小 Schema + 指纹，见附录 A）
        request_id = str(uuid.uuid4())
        rng_state = f"py_random_seed={args.seed}"
        prewarm_state = "cold" if warm_left > 0 else "warm"
        right_censored = bool(ret.get("right_censored", False))

        in_tokens = count_tokens(prompt, mode=args.tokenizer)
        out_tokens = int(ret.get("eval_tokens", 0))
        billing_breakdown = {
            "in_tokens": float(in_tokens), "out_tokens": float(out_tokens),
            "prefill_ms": None, "kv_ms": None, "overhead_ms": None
        }

        row = {
            "id": idx,
            "mode": args.mode,
            "session_id": session_id,
            "request_id": request_id,
            "user_tenant": args.user_tenant,
            "prompt": prompt,
            "_prompt_obj": obj,
            "output": ret.get("text",""),
            "prompt_len": len(prompt or ""),
            "need_json": need_json,
            "plan_complexity": plan_report.get("complexity"),
            "route_decision": route_decision,
            "final_path": ret["final_path"],
            "attempts": ret["attempts"],
            "upgrades": ret.get("upgrades", 0),
            "repaired": ret.get("repaired", False),
            "retry_count": max(0, ret["attempts"] - 1),
            "tool_retry": max(0, ret["attempts"] - 1),
            "upgrade_reason": ret["upgrade_reason"],
            "json_ok": json_ok,
            "json_violation": json_violation,
            "e2e_ms": ret["e2e_ms"],
            "eval_tokens": out_tokens,
            "in_warmup": warm_left > 0,
            "refusal": is_refusal(ret.get("text","")),
            "error": not ok,
            "error_msg": ret.get("error", ""),
            "error_kind": ret.get("error_kind", ""),

            # 删失字段
            "right_censored": right_censored,

            # 系统/审计指纹
            "rng_state": rng_state,
            "seed": args.seed,
            "batch_size_eff": 1,
            "prewarm_state": prewarm_state,
            "init_ms": None,
            "region": args.region,
            "az": args.az,
            "network_rtt_ms": None,
            "queue_ms": 0.0,
            "cache_hit": {"prompt": False, "kv": False},
            "kv_pages": None, "kv_hit": None, "paged_faults": None,
            "prefill_share": None, "kv_share": None,
            "route": ret["final_path"],
            "config_fingerprint": f"model={ret['final_path']},temp={args.temperature},topp={args.top_p},ctx={args.num_ctx}",
            "billing_version": BILLING_VERSION,
            "billing_breakdown": billing_breakdown,
            "gc_events": 0,
            "co_correction": (args.co_correction != "none"),
            "tdigest_compression": int(args.tdigest_compression),
            "tdigest_version": TDIGEST_VERSION or "unknown",
            "warmup_window_ms": int(args.warmup_window_ms),
        }
        rows.append(row)
        if warm_left > 0:
            warm_left -= 1

    # 汇总与四口径
    stats = compute_stats_and_summary(rows, args, tdigest_version=TDIGEST_VERSION)
    summary = stats.summary
    warm_df = stats.warm_df
    df_all = stats.df_all

    # 与最近 baseline 比较（主判 CO warm）
    base_sum_path = latest(os.path.join("runs", "baseline_enh_*_summary.json"))
    base_events_path = latest(os.path.join("runs", "baseline_enh_*_events.jsonl"))

    if args.mode == "gateway_3stage" and base_sum_path:
        with open(base_sum_path, "r", encoding="utf-8-sig") as f:
            bsum = json.load(f)
        b95 = bsum.get("warm_p95_ms_co", bsum.get("warm_p95_ms", np.nan))
        try:
            b95 = float(b95)
        except Exception:
            b95 = np.nan
        g95 = float(summary["warm_p95_ms_co"])
        if not (np.isnan(b95) or np.isnan(g95)):
            delta = 100.0 * (g95 - b95) / max(b95, EPS)
            summary["p95_guardrail_vs_baseline_pct"] = round(delta, 2)
            summary["p95_guardrail_pass_co"] = (delta <= 100.0 * args.p95_delta)
        if "warm_p95_ms" in bsum:
            g95_raw = float(summary["warm_p95_ms"])
            b95_raw = float(bsum["warm_p95_ms"])
            d_raw = 100.0 * (g95_raw - b95_raw) / max(b95_raw, EPS)
            summary["p95_guardrail_vs_baseline_pct_raw"] = round(d_raw, 2)
            summary["p95_guardrail_pass_raw"] = (d_raw <= 100.0 * args.p95_delta)

    # 家族检验（A/B/C）
    tests_A: List[Dict[str, Any]] = []
    tests_B: List[Dict[str, Any]] = []
    tests_C: List[Dict[str, Any]] = family_C_json_guard(warm_df, q=args.fdr_q)

    base_warm_df: Optional[pd.DataFrame] = None
    if base_events_path is not None and os.path.exists(base_events_path):
        base_events = pd.read_json(base_events_path, lines=True)
        base_warm_df = base_events[~base_events["in_warmup"]].copy()

    # 家族 B：CO（warm）单侧护栏
    if args.mode == "gateway_3stage" and base_warm_df is not None and not base_warm_df.empty:
        tests_B = family_B_p95_guardrail(warm_df, base_warm_df, args)

    # 家族 A：TOST 占位（如无质量标签则 N/A）
    tests_A = family_A_tost_placeholder(warm_df, base_warm_df, epsilon_q=args.epsilon_q)

    # 落盘（含 bootstrap_meta 与 seed_manifest）
    write_outputs(df_all, summary, tests_A, tests_B, tests_C, args, censoring_mode=summary.get("censoring_mode","KM / SLO-cap"))

    # 控制台打印
    print("[MODE]", args.mode)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
