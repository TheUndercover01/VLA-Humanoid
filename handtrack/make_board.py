"""Render the ChArUco board as an A4 PNG at 300 dpi.

Print at 100% / "actual size" (no fit-to-page), then measure one square with a
ruler and use that value as --square-mm everywhere else.

    python -m handtrack.make_board --out board_a4.png
"""
import argparse

import cv2
import numpy as np

from handtrack.board import SQUARES_X, SQUARES_Y, SQUARE_MM, make_board

DPI = 300
A4_MM = (210.0, 297.0)


def mm_to_px(mm):
    return int(round(mm / 25.4 * DPI))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="board_a4.png")
    args = ap.parse_args()

    board = make_board()
    sq_px = mm_to_px(SQUARE_MM)
    board_img = board.generateImage((SQUARES_X * sq_px, SQUARES_Y * sq_px), marginSize=0)

    page = np.full((mm_to_px(A4_MM[1]), mm_to_px(A4_MM[0])), 255, np.uint8)
    y0 = (page.shape[0] - board_img.shape[0]) // 2
    x0 = (page.shape[1] - board_img.shape[1]) // 2
    page[y0:y0 + board_img.shape[0], x0:x0 + board_img.shape[1]] = board_img

    # 100 mm scale bar to check the print scale
    bar_y = y0 + board_img.shape[0] + mm_to_px(8)
    cv2.line(page, (x0, bar_y), (x0 + mm_to_px(100), bar_y), 0, 4)
    cv2.putText(page, "100 mm - check with a ruler", (x0, bar_y + mm_to_px(6)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, 0, 2)

    cv2.imwrite(args.out, page)
    print(f"wrote {args.out}: {SQUARES_X}x{SQUARES_Y} squares, {SQUARE_MM} mm nominal")


if __name__ == "__main__":
    main()
