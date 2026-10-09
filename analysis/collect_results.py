"""Collect every evaluation line of cache/distill_status.txt into one long table (results/comparative/evals_long.csv).

    python analysis/collect_results.py [--status /media/storage/ayush/cache/distill_status.txt] [--out results/comparative/evals_long.csv]

One row per (model, setting, chain): strict success (older evals only), lenient, in-order, commands done in order, and the raw line.
The formats of the log changed over the week (strict + lenient, then lenient + in-order, then done-flag settings); lines that match
none of them stay in the raw copy of the log (results/comparative/logs/distill_status.txt).
"""
import argparse
import csv
import re

PATTERNS = [
    # VLA <model> [<setting>] <chain>: success S/100, ... lenient L/100  (older: strict + lenient)
    ("strict_lenient", re.compile(r"^(?P<t>\w{3} \d\d:\d\d) VLA (?P<model>\S+) \[(?P<setting>[^\]]+)\] (?P<chain>\w+): success (?P<strict>\d+)/100.*?lenient (?P<lenient>\d+)/100")),
    # VLA <model> [<setting>] <chain>: lenient L/100, in-order O/100 (mean commands in order X/N)
    ("lenient_inorder", re.compile(r"^(?P<t>\w{3} \d\d:\d\d) VLA (?P<model>\S+) \[(?P<setting>[^\]]+)\] (?P<chain>\w+): lenient (?P<lenient>\d+)/100, in-order (?P<inorder>\d+)/100(?: \(mean commands in order (?P<cmds>[\d.]+)/(?P<n>\d+)\))?")),
    # VLA <model> [<setting>] <chain>: lenient L/100  (no in-order)
    ("lenient_only", re.compile(r"^(?P<t>\w{3} \d\d:\d\d) VLA (?P<model>\S+) \[(?P<setting>[^\]]+)\] (?P<chain>\w+): lenient (?P<lenient>\d+)/100\s*$")),
    # ck32k whole push lenient 32 in-order 32 (0.32/1)
    ("ck_custom", re.compile(r"^(?P<t>\w{3} \d\d:\d\d) VLA (?P<model>\S+) (?P<setting>whole|step) (?P<chain>\w+) lenient (?P<lenient>\d+) in-order (?P<inorder>\d+) \((?P<cmds>[\d.]+)/(?P<n>\d+)\)")),
    # sweep <model> done_thresh T hold H <chain>: lenient L/100, in-order O/100 (mean commands in order X/N)
    ("sweep", re.compile(r"^(?P<t>\w{3} \d\d:\d\d) sweep (?P<model>\S+) done_thresh (?P<th>[\d.]+) hold (?P<hold>\d+) (?P<chain>\w+): lenient (?P<lenient>\d+)/100, in-order (?P<inorder>\d+)/100 \(mean commands in order (?P<cmds>[\d.]+)/(?P<n>\d+)\)")),
    # teacher <run> model_N [<chain>]: lenient L/100, in-order O/100
    ("teacher", re.compile(r"^(?P<t>\w{3} \d\d:\d\d) teacher (?P<model>\S+) (?P<ck>model_\d+) \[(?P<chain>\w+)\]: lenient (?P<lenient>\d+)/100, in-order (?P<inorder>\d+)/100")),
]
FIELDS = ["time", "kind", "model", "setting", "chain", "strict", "lenient", "in_order", "cmds_in_order", "n_commands", "raw"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", default="/media/storage/ayush/cache/distill_status.txt")
    ap.add_argument("--out", default="results/comparative/evals_long.csv")
    a = ap.parse_args()
    rows, skipped = [], 0
    for line in open(a.status, errors="replace"):
        line = line.rstrip("\n")
        for kind, pat in PATTERNS:
            m = pat.match(line)
            if m:
                g = m.groupdict()
                setting = g.get("setting") or (f"done_thresh {g['th']} hold {g['hold']}" if "th" in g else g.get("ck", ""))
                kind_out = "teacher" if kind == "teacher" else kind
                rows.append({"time": g["t"], "kind": kind_out, "model": g["model"], "setting": setting, "chain": g["chain"],
                             "strict": g.get("strict", ""), "lenient": g["lenient"], "in_order": g.get("inorder", ""),
                             "cmds_in_order": g.get("cmds", "") or "", "n_commands": g.get("n", "") or "", "raw": line[:300]})
                break
        else:
            skipped += 1
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} evaluation rows -> {a.out} ({skipped} other lines stay in the raw log)")


if __name__ == "__main__":
    main()
