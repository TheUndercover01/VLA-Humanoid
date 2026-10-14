"""14 Oct night: waits (detached) for the clip-tolerance pipeline (CLIPTOL) to end, done or failed, then writes MORNING_REPORT.md (+ PROGRESS.md):
the teacher's check, the label heights, the new student's chains next to the earlier students, the re-scoring with the teacher's tolerance
(RESCORE), the mark positions (MARKPOS) and the tic-tac-toe student (TTT)."""
import re
import time
from pathlib import Path

ST = Path("/media/storage/ayush/cache/distill_status.txt")
end = lambda t: re.search(r"CLIPTOL .*(pipeline done|check failed|ended early|failed)", t)
t0 = time.time()
while not end(ST.read_text(errors="ignore")) and time.time() - t0 < 16 * 3600:
    time.sleep(120)
time.sleep(60)
txt = ST.read_text(errors="ignore")
lines = [l for l in txt.splitlines() if l[:3] in ("Wed", "Thu")]
pick = lambda tag: [l for l in lines if tag in l]
out = ["## Morning report, 14 Oct: clip-derived put-down tolerance, teacher from scratch, student, re-scoring", ""]
out += ["### Clip-tolerance pipeline (CLIPTOL)", ""] + [f"- {l}" for l in pick("CLIPTOL") if "VLA " not in l] + [""]
pat = r"VLA (\S+) \[[^\]]*\] (\S+): lenient (\d+)/100, in-order (\d+)/100 \(mean commands in order ([0-9.]+)/(\d+)\)"
rows = {}
for l in lines:
    m = re.search(pat, l)
    if m and m.group(1) in ("ours_cyl_marks_cliptol", "ours_cyl_marks_handleg_rand", "ours_cyl_marks_nolift", "ours_cyl_marks_progress"):
        rows.setdefault(m.group(2), {})[m.group(1)] = f"{m.group(3)} / {m.group(4)} ({m.group(5)}/{m.group(6)})"
names = ["ours_cyl_marks_cliptol", "ours_cyl_marks_handleg_rand", "ours_cyl_marks_nolift", "ours_cyl_marks_progress"]
out += ["### Students, lenient / in order (mean commands in order); the new one scored with the clip tolerance, the others with the old 1.5 cm", "",
        "| chain | " + " | ".join(n.replace("ours_cyl_marks_", "") for n in names) + " |", "|---" * (len(names) + 1) + "|"]
for ch in ["push", "lift", "lift_cyl", "c1", "c1_cyl", "c2", "swap", "unstack", "restack", "restack8", "rand8_1", "rand8_2"]:
    if ch in rows:
        out.append(f"| {ch} | " + " | ".join(rows[ch].get(n, "-") for n in names) + " |")
out += ["", "### Re-scoring of the earlier students with the teacher's own tolerance (RESCORE)", ""] + [f"- {l}" for l in pick("RESCORE")]
out += ["", "### Where the student's marks really are (MARKPOS)", ""] + [f"- {l}" for l in pick("MARKPOS")]
tab = Path("/media/storage/ayush/cache/markpos_table.txt")
if tab.exists():
    out += ["", "```", tab.read_text().strip(), "```"]
out += ["", "### Tic-tac-toe student (TTT)", ""] + [f"- {l}" for l in pick("TTT")[-10:]]
rep = "\n".join(out) + "\n"
Path("MORNING_REPORT.md").write_text(rep)
with open("PROGRESS.md", "a") as f:
    f.write("\n" + rep)
