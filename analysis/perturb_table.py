"""Perturbation study table: per run, scenes disturbed and redone, split by how many "(done)" marks the policy had written at the
disturbance (runs/eval/perturb/*.csv from scripts/perturb_study.sh).   python analysis/perturb_table.py [csv ...]"""
import csv
import glob
import sys

files = sys.argv[1:] or sorted(f for f in glob.glob("runs/eval/perturb/*.csv") if "smoke" not in f)
for f in files:
    r = list(csv.DictReader(open(f)))
    name = f.split("/")[-1][:-4]
    line = f"{name:40s} n={len(r)} lenient {sum(int(x['lenient']) for x in r)} in order {sum(int(x['ordered']) for x in r)}"
    if "disturbed" in r[0]:
        d = [x for x in r if int(x["disturbed"])]
        by = {}
        for x in d:
            k = x["marks_at_disturb"]
            by.setdefault(k, [0, 0])
            by[k][0] += 1
            by[k][1] += int(x["recovered"])
        line += f" | disturbed {len(d)}, redone {sum(int(x['recovered']) for x in d)} | by marks at the disturbance " + \
                ", ".join(f"{k}: {v[1]}/{v[0]}" for k, v in sorted(by.items()))
    print(line)
