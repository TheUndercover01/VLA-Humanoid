"""12 Oct night: waits (detached) for the random-start student's 10-chain eval, then writes MORNING_REPORT.md and appends it to PROGRESS.md.
Compares ours_cyl_marks_handleg_rand with the earlier hand-leg student and the headline (realref), lenient / in order, from distill_status.txt."""
import re, subprocess, time
from pathlib import Path

C = Path("/media/storage/ayush/cache"); ST = C / "distill_status.txt"
NEW, CHAINS = "ours_cyl_marks_handleg_rand", "push lift lift_cyl c1 c1_cyl c2 c3 swap unstack restack".split()
done = lambda t: re.search(re.escape(NEW) + r" \[WHOLE[^\]]*\] eval done", t) is not None
t0 = time.time()
while not done(ST.read_text(errors="ignore")):
    if time.time() - t0 > 14 * 3600:
        break
    time.sleep(60)
txt = ST.read_text(errors="ignore")
pat = r"lenient (\d+)/100, in-order (\d+)/100"

def rows(prefix_re):
    out = {}
    for ch in CHAINS:
        m = re.findall(prefix_re.format(ch=ch) + r".*?" + pat, txt)
        out[ch] = f"{m[-1][0]} / {m[-1][1]}" if m else "-"
    return out

new = rows(r"VLA " + NEW + r" \[WHOLE[^\]]*\] {ch}:")
old = rows(r"VLA ours_cyl_marks_handleg \[WHOLE[^\]]*\] {ch}:")
ref = rows(r"REALREF ours_cyl_marks_realref {ch}:")
ft = re.findall(rf"fine-tuned {NEW}: (.*)", txt)
lines = ["## Mon 12 / Tue 13 Oct — random-start student `" + NEW + "` (hand-set thresholds, hand leg, random start pose, padding 10), 10 chains x 100 val layouts, lenient / in order", "",
         f"- Training: {ft[-1] if ft else 'not finished (see the status log)'}", "",
         "| chain | new (random start, pad 10) | hand-leg student (pad 20) | headline realref |", "|---|---|---|---|"]
lines += [f"| {ch} | {new[ch]} | {old[ch]} | {ref[ch]} |" for ch in CHAINS]
lines += ["", "(read the status lines `HANDLEG_RAND` and `VLA " + NEW + "` in cache/distill_status.txt for the raw rows; a '-' = chain not evaluated for that student)"]
rep = "\n".join(lines) + "\n"
Path("MORNING_REPORT.md").write_text(rep)
with open("PROGRESS.md", "a") as f:
    f.write("\n" + rep)
