"""Picks, per chain, in-order SUCCESSES and FAILURES (end result reached but not in order / not reached) from the eval csv of a
scripts/headline_videos.sh run, copies the videos to --out, makes 8-frame contact sheets and writes a report (what the csv says).
14 Oct: first used for ours_cyl_marks_progress (c3 / swap / unstack); now takes the student, chains and output as arguments.
    python3 analysis/pick_videos.py --name ours_cyl_marks_nolift --chains restack unstack restack8 --out media/headline/final --k 2"""
import argparse
import csv
import shutil
import subprocess
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--name", default="ours_cyl_marks_progress")
ap.add_argument("--chains", nargs="+", default=["c3", "swap", "unstack"])
ap.add_argument("--out", default="media/headline/pick")
ap.add_argument("--report", default="VIDEOS_REPORT.md")
ap.add_argument("--k", type=int, default=1, help="videos per group")
ap.add_argument("--wait", action="store_true", help="wait for 'HEADLINE <name> done' in the status log first")
args = ap.parse_args()
C = Path("/media/storage/ayush/cache/distill_status.txt")
NAME = args.name
t0 = time.time()
while args.wait and f"HEADLINE {NAME} done" not in C.read_text(errors="ignore") and time.time() - t0 < 4 * 3600:
    time.sleep(30)
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
lines = [f"# Videos of `{NAME}`: {', '.join(args.chains)}", ""]
sheets = []
for ch in args.chains:
    f = Path(f"runs/eval/headline/{NAME}_{ch}.csv")
    if not f.exists():
        lines.append(f"## {ch}: no csv ({f})")
        continue
    rows = list(csv.DictReader(open(f)))
    ok = [r for r in rows if int(float(r["ordered"]))]
    lenient_only = [r for r in rows if not int(float(r["ordered"])) and int(float(r["lenient"]))]
    bad = [r for r in rows if not int(float(r["lenient"]))]
    lines += [f"## {ch}: in order {len(ok)}/{len(rows)}, end result reached but not in order {len(lenient_only)}, end result not reached {len(bad)}", ""]
    for tag, grp in (("SUCCESS (all commands in order)", ok[:args.k]), ("FAILURE A: end result reached, commands NOT in order", lenient_only[:args.k]),
                     ("FAILURE B: end result not reached", bad[:args.k])):
        for r in grp:
            i = int(float(r["state"]))
            src = Path(f"media/headline/{NAME}_{ch}_env{i}.mp4")
            if not src.exists():
                lines.append(f"- {tag}: env {i}: video missing ({src})")
                continue
            dst = out / f"{ch}_{tag.split(':')[0].split(' ')[0].lower()}{'_A' if 'A:' in tag else '_B' if 'B:' in tag else ''}_env{i}.mp4"
            shutil.copy(src, dst)
            sheets.append(str(dst))
            lines.append(f"- **{tag}** -> `{dst}`: commands done in order {r['ordered_done']}/{r['n_commands']}, failed at `{r['failed_at'] or '-'}`, "
                         f"cube rose {float(r['first_rise_cm']):.0f} cm, cylinder rose {float(r['other_rise_cm']):.0f} cm, {float(r['time']):.0f} s")
    lines.append("all runs: " + ", ".join(f"env {int(float(r['state']))} {r['ordered_done']}/{r['n_commands']} {r['failed_at'] or 'ok'}" for r in rows))
    lines.append("")
if sheets:
    subprocess.run(["python3", "analysis/video_sheets.py", *sheets, "--out", str(out / "sheets")])
lines.append(f"Contact sheets (8 frames per video): `{out}/sheets/`. The model samples noise, so the videos are the runs the csv rows describe (same eval run).")
Path(args.report).write_text("\n".join(lines) + "\n")
