import sys, json, csv, argparse

ap = argparse.ArgumentParser()
ap.add_argument("--in", dest="infile", required=True)
ap.add_argument("--acc-out", dest="acc_out", required=True)
ap.add_argument("--json-out", dest="json_out", required=True)
args = ap.parse_args()

def is_json_ok(s: str) -> bool:
    s = (s or "").strip()
    if not s: return False
    a,b = s.find("{"), s.rfind("}")
    c,d = s.find("["), s.rfind("]")
    cands=[s]
    if a!=-1 and b!=-1 and b>a: cands.append(s[a:b+1])
    if c!=-1 and d!=-1 and d>c: cands.append(s[c:d+1])
    for seg in cands:
        try: json.loads(seg); return True
        except Exception: pass
    return False

rows=[]
with open(args.infile,"r",encoding="utf-8-sig") as f:
    for i,ln in enumerate(f,1):
        ln=ln.strip()
        if not ln: continue
        obj=json.loads(ln)
        rid = obj.get("id") if isinstance(obj,dict) else i
        out = ""
        if isinstance(obj,dict):
            out = str(obj.get("output","") or obj.get("text","") or "")
        s_acc  = 1.0 if out.strip()!="" else 0.0
        s_json = 1.0 if is_json_ok(out) else 0.0
        rows.append((int(rid), s_acc, s_json))

# TOST 需要四列：score,id,metric,stratum（顺序无所谓，列名必须有）
with open(args.acc_out,"w",newline="",encoding="utf-8") as f:
    w=csv.writer(f); w.writerow(["score","id","metric","stratum"])
    for rid,s_acc,s_json in rows:
        w.writerow([s_acc, rid, "acc", "general"])

with open(args.json_out,"w",newline="",encoding="utf-8") as f:
    w=csv.writer(f); w.writerow(["score","id","metric","stratum"])
    for rid,s_acc,s_json in rows:
        w.writerow([s_json, rid, "json", "json"])

print(f"wrote: {args.acc_out} ; {args.json_out}")
