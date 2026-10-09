"""Cut one long spoken-label recording into one clip per skill (FILMING.md).

While filming, say the label, then "start", hold still ~1 s, do the skill, hold still ~1 s, say "end":

    "pick red. start."   ...lift...   "end."
    "place blue stack. start."   ...   "end."          (blue ends up on the red cube)
    "place red target. start."   ...   "end."
    "push blue. start."   ...   "end."
    "board. start." ... "end."                         (board alone, flat, ~5 s)
    "calibrate. start." ... "end."                     (board tilted and moved, ~20 s)
    "cancel."                                          (drops the clip you are in the middle of / just did)

Whisper gives word times; a clip runs from the end of the word "start" to the start of the word "end", so the
spoken words are not in it (the still seconds absorb the +-0.2 s of word timing).

    /media/storage/ayush/envs/asr/bin/python -m handtrack.cut_session session.mp4 --out data/raw/clips [--dry_run]

Writes <out>/<skill>_<NN>[_stack|_target].mp4 (colour is in segments.csv: the tracker gets the cube from the ArUco id
anyway) and <out>/segments.csv (file, label, cube, t0, t1, words). Check segments.csv before using the clips.
"""
import argparse
import csv
import difflib
import re
import subprocess
from pathlib import Path

import numpy as np

SKILL = {"pick": "pick-lift", "lift": "pick-lift", "push": "push", "place": "place-down", "drop": "place-down",
         "stack": "place-down", "stacking": "place-down"}
OBJECTS = ("red", "blue", "cube", "cylinder")
WORDS = ["start", "end", "cancel", "sorry", "pick", "lift", "push", "place", "down", "drop", "stack", "stacking", "on", "onto",
         "red", "blue", "cube", "cylinder", "target", "board", "calibrate"]
ALIAS = {"pic": "pick", "pig": "pick", "plays": "place", "plate": "place", "pitch": "push", "bush": "push", "read": "red",
         "blew": "blue", "star": "start", "stark": "start", "tart": "start", "calibrated": "calibrate", "calibration": "calibrate",
         "cue": "cube", "cubes": "cube", "cylinders": "cylinder", "cylindr": "cylinder", "handle": "cylinder"}
# file-name defaults (6 Oct: one label said at the top of a file, then many "start ... end" takes without repeating it;
# the spoken label is often misheard, so the file name <Skill>_<object>.MOV is the fallback)
FILE_LABELS = {"push_cube": "push cube", "push_cylinder": "push cylinder", "lift_cube": "pick cube", "lift_cylinder": "pick cylinder",
               "cube_lift": "pick cube", "cylinder_lift": "pick cylinder", "cube_drop": "place down cube", "cylinder_drop": "place down cylinder",
               "cube_cylinder": "stack cube on cylinder", "cylinder_cube": "stack cylinder on cube"}


def audio(path, sr=16000):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32)


PROMPT = "push cube. place down cylinder. cube on cylinder. pick red. board. calibrate. start. end. cancel."   # biases Whisper to the vocabulary


def transcribe(wav, model, device):
    from faster_whisper import WhisperModel      # in /media/storage/ayush/envs/asr (transformers' pipeline needs torchcodec)
    m = WhisperModel(model, device=device, compute_type="int8" if device == "cpu" else "float16")
    segs, _ = m.transcribe(wav, language="en", word_timestamps=True, initial_prompt=PROMPT,
                           condition_on_previous_text=False, beam_size=5)
    return [(w.word, w.start, w.end) for sg in segs for w in sg.words]


def normalise(word, inside=False):
    w = re.sub(r"[^a-z]", "", word.lower())
    if inside and w in ("and", "n", "in", "ends"):        # "end" is often heard as "and" at the end of a take
        return "end"
    if w in ALIAS:
        return ALIAS[w]
    hit = difflib.get_close_matches(w, WORDS, n=1, cutoff=0.8)
    return hit[0] if hit else None


def segments(words, default_label=()):
    """words: [(text, t0, t1)] -> [{label, t0, t1, flag}]. The label (spoken once, before the first "start") applies to
    every following take until new label words are spoken. "start" while a take is open restarts it (its "end" was
    missed); "cancel"/"sorry" inside a take drops it."""
    out, label, inside, t_in, fresh, spoken = [], [], False, None, False, False
    for text, a, b in words:
        w = normalise(text, inside)
        if w is None:
            continue
        if not inside:
            if w == "start":
                inside, t_in = True, b
            elif w in ("cancel", "sorry", "end"):
                continue
            else:
                if fresh:
                    label, fresh = [], False
                label.append(w)
        else:
            if w == "start":                                 # the previous "end" was missed: this is a new take
                out.append({"label": label or list(default_label), "t0": t_in, "t1": None, "flag": "no end heard"})
                t_in = b
            elif w == "end":
                out.append({"label": label or list(default_label), "t0": t_in, "t1": a, "flag": ""})
                inside, fresh = False, True
                if any(x in ("calibrate", "board") for x in label):   # the file's own skill comes next (6 Oct, Cube_lift)
                    label, fresh = [], False
            elif w in ("cancel", "sorry"):
                inside, fresh = False, True
    return out


def describe(seg):
    """label words -> (skill, object, destination) or None. 'stack cube on cylinder' -> (place-down, cube, cylinder);
    'place down cube' -> (place-down, cube, table); 'push cube' -> (push, cube, target)."""
    lab = seg["label"]
    if "board" in lab:
        return "board", "", ""
    if "calibrate" in lab:
        return "calibrate", "", ""
    skill = next((SKILL[w] for w in lab if w in SKILL), None)
    objs = [w for w in lab if w in OBJECTS]
    if skill is None and len(objs) == 2 and ("on" in lab or "onto" in lab):
        skill = "place-down"
    if skill is None or not objs:
        return None
    if skill == "place-down":
        dest = objs[1] if len(objs) > 1 else ("target" if "target" in lab else "table")
    else:
        dest = "target" if skill == "push" else ""
    return skill, objs[0], dest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="small.en")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--label", default="", help="label words for every take of this file, e.g. 'stack cube on cylinder' "
                    "(default: from the file name, e.g. Cube_cylinder.MOV, then what you said at the start)")
    ap.add_argument("--min_take_s", type=float, default=1.5, help="shorter takes are mis-heard words, skipped")
    ap.add_argument("--max_take_s", type=float, default=30, help="takes longer than this are flagged")
    ap.add_argument("--dry_run", action="store_true", help="print the transcript and the segments, cut nothing")
    a = ap.parse_args()
    stem = Path(a.video).stem
    default = (a.label or FILE_LABELS.get(stem.lower(), "")).split()
    words = transcribe(audio(a.video), a.model, a.device)
    print("transcript:", " ".join(f"{t.strip()}@{s:.1f}" for t, s, _ in words))
    segs = segments(words, default)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    count, rows = {}, []
    for s in segs:
        flag = s["flag"]
        if s["t1"] is None:
            print(f"skipping take at {s['t0']:.1f} s: no 'end' heard")
            continue
        if s["t1"] - s["t0"] < a.min_take_s:
            print(f"skipping {s['t0']:.1f}-{s['t1']:.1f} s: shorter than {a.min_take_s} s")
            continue
        d = describe(s)
        if d is None and default:
            d = describe({"label": default})                 # the spoken label was misheard: the file name decides
            flag = (flag + " label from file name").strip()
        if d is None:
            print(f"skipping {s['t0']:.1f}-{s['t1']:.1f} s: label not understood ({' '.join(s['label'])!r})")
            continue
        skill, obj, dest = d
        if s["t1"] - s["t0"] > a.max_take_s:
            flag = (flag + " long take").strip()
        key = f"{skill}_{obj}_{dest}".strip("_")
        count[key] = count.get(key, 0) + 1
        name = f"{stem}_{count[key]:02d}.mp4" if skill not in ("calibrate", "board") else f"{stem}_{skill}_{count[key]:02d}.mp4"
        rows.append([name, skill, obj, dest, f"{s['t0']:.2f}", f"{s['t1']:.2f}", flag])
        print(f"{name}: {skill} {obj} {dest} {s['t0']:.1f} -> {s['t1']:.1f} s ({s['t1'] - s['t0']:.1f} s) {flag}")
        if not a.dry_run:
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", a.video, "-ss", f"{s['t0']:.3f}", "-to", f"{s['t1']:.3f}",
                            "-c:v", "libx264", "-crf", "17", "-an", str(out / name)], check=True)
    if not a.dry_run:
        with open(out / f"segments_{stem}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["file", "skill", "object", "dest", "t0", "t1", "flag"])
            w.writerows(rows)
    print(f"{len(rows)} clips")


if __name__ == "__main__":
    main()
