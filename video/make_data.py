"""results.json (PC) + anatomy.json -> public/data.json, the only numbers the video shows."""
import json, sys
r = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "results.json"))
a = json.load(open("public/anatomy.json"))
S = r["summary"]
hero = next(c for c in r["cases"] if c["case"] == "s1922")
data = {"hero": {"case": "s1922", "seconds": hero["trt_s"], "found": len(a["structures"]), "ct_slices": 421},
        "eval": {"cases": S["n_cases"], "median_dice": S["median_dice"], "structure_checks": S["structure_checks"],
                 "pct_above_090": round(S["pct_above_090"]), "median_seconds": S["trt_s"]["median"],
                 "min_seconds": S["trt_s"]["min"], "max_seconds": S["trt_s"]["max"]},
        "repo": "github.com/zalkarbek2023/ct-anatomy-3d"}
json.dump(data, open("public/data.json", "w"), indent=1)
print(json.dumps(data, indent=1))
