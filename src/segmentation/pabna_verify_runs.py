"""Integrity check of all downloaded Kaggle runs (A..J).
Checks per run: training reached total_steps, tuning.json written before TEST_DONE.flag,
tuning sha256 consistent, threshold chosen on validation only, one prediction mask per test tile,
per-tile confusion sums equal test_summary.json, expected seed, split and test tile count.
Usage: python -I pabna_verify_runs.py <archive_dir>
"""
import csv, glob, hashlib, json, os, sys

arch = sys.argv[1]
EXPECT_TILES = {"S2_fold1": 485, "S2_fold2": 442, "S2_fold3": 376, "S1": 131}
rows, problems = [], []
for sess in "ABCDEFGHIJ":
    for rd in sorted(glob.glob(os.path.join(arch, f"kaggle_train_{sess}", "runs", "*"))):
        if not os.path.isdir(rd):
            continue
        rid = os.path.basename(rd)
        J = lambda f: os.path.join(rd, f)
        cfg = json.load(open(J("config.json")))
        summ = json.load(open(J("test_summary.json")))
        tun = json.load(open(J("tuning.json")))
        flag = open(J("TEST_DONE.flag")).read().strip()
        last = list(csv.DictReader(open(J("train_log.csv"))))[-1]
        steps_ok = int(last["step"]) == cfg["total_steps"] == 69600
        tune_first = tun["created"] < flag
        sha_in = tun.pop("sha256", None)
        sha_alt = hashlib.sha256(json.dumps(tun, sort_keys=True).encode()).hexdigest()
        tun["sha256"] = sha_in
        valid_only = tun.get("chosen_on") == "validation subset only"
        thr_match = abs(tun["threshold"] - summ["threshold"]) < 1e-9 and tun["A_min_m2"] == summ["A_min_m2"]
        pt = list(csv.DictReader(open(J("per_tile_test.csv"))))
        sums = {k: sum(int(r[k]) for r in pt) for k in ("TP", "FP", "FN", "TN")}
        sums_ok = all(sums[k] == summ[k] for k in sums)
        masks = len(glob.glob(J("test_pred_masks/*.png")))
        split = cfg["split"]
        fold = int(split[-1]) if split.startswith("S2_fold") else 0
        seed_exp = 20261106 + 1000 * fold + cfg["seed_idx"]
        tiles_ok = summ["n_tiles"] == len(pt) == masks == EXPECT_TILES.get(split, -1)
        checks = dict(steps=steps_ok, tune_before_test=tune_first, valid_only=valid_only,
                      thr_match=thr_match, sums=sums_ok, tiles_masks=tiles_ok,
                      seed=cfg["seed"] == seed_exp, not_smoke=not cfg["smoke"] and not summ["smoke"])
        bad = [k for k, v in checks.items() if not v]
        if bad:
            problems.append((sess, rid, bad))
        rows.append(dict(session=sess, run=rid, tiles=summ["n_tiles"], F1=round(summ["F1"], 4),
                         IoU=round(summ["IoU"], 4), P=round(summ["precision"], 4), R=round(summ["recall"], 4),
                         objF1=round(summ["obj_F1"], 4), best_step=summ["best_step"], thr=summ["threshold"],
                         Amin=summ["A_min_m2"], tuned=tun["created"], test_done=flag[:19],
                         sha256_recomputed_match=(sha_alt == sha_in), all_checks_ok=not bad))
out = os.path.join(arch, "verify_all_runs.csv")
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)
for r in rows:
    print(r["session"], f'{r["run"]:22s}', r["tiles"], r["F1"], r["IoU"], r["P"], r["R"], r["objF1"],
          r["best_step"], r["thr"], r["Amin"], "OK" if r["all_checks_ok"] else "FAIL", "sha" if r["sha256_recomputed_match"] else "sha?")
print(f"\nruns: {len(rows)}; problems: {problems if problems else 'none'}\nwrote {out}")
