"""Render the cube and pad ArUco markers as an A4 PNG at 300 dpi (same dictionary as the board).

IDs: 20 = red cube, 21 = blue cube, 22 = target pad (the board uses 0-16). Three copies of each cube marker (top
face + spares for the sides), two of the pad. Print at 100% / "actual size", cut each out keeping the white border,
glue flat on the cube's top face (centred, edges parallel to the cube's edges), measure one marker with a ruler and
note it in notes.txt.

    python -m handtrack.make_markers --out markers_a4.png [--marker-mm 40]
"""
import argparse

import cv2
import numpy as np

from handtrack.board import DICT
from handtrack.make_board import A4_MM, mm_to_px

IDS = [20, 20, 20, 21, 21, 21, 22, 22]
NAMES = {20: "red cube", 21: "blue cube", 22: "target pad"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="markers_a4.png")
    ap.add_argument("--marker-mm", type=float, default=40.0, help="side of the black square (cube faces are 50 mm)")
    a = ap.parse_args()
    dictionary = cv2.aruco.getPredefinedDictionary(DICT)
    m, cell = mm_to_px(a.marker_mm), mm_to_px(a.marker_mm + 14)
    page = np.full((mm_to_px(A4_MM[1]), mm_to_px(A4_MM[0])), 255, np.uint8)
    for i, mid in enumerate(IDS):
        r, c = divmod(i, 3)
        y0, x0 = mm_to_px(15) + r * cell, mm_to_px(15) + c * cell
        img = cv2.aruco.generateImageMarker(dictionary, mid, m)
        o = (cell - m) // 2
        page[y0 + o:y0 + o + m, x0 + o:x0 + o + m] = img
        cv2.rectangle(page, (x0, y0), (x0 + cell, y0 + cell), 200, 1)               # cut guide, outside the marker
        cv2.putText(page, f"id {mid} {NAMES[mid]}", (x0 + 6, y0 + cell - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 120, 1)
    y = mm_to_px(15) + 3 * cell + mm_to_px(8)
    cv2.line(page, (mm_to_px(15), y), (mm_to_px(115), y), 0, 4)
    cv2.putText(page, "100 mm - check with a ruler", (mm_to_px(15), y + mm_to_px(6)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 0, 2)
    cv2.imwrite(a.out, page)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
