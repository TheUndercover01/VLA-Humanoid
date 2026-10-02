"""Shared ChArUco board definition.

One printed board does two jobs: camera calibration and a metric table frame
in every video frame. Table frame = board frame: origin at the board's corner,
x/y along the board edges, z up out of the table.
"""
import cv2

SQUARES_X = 5
SQUARES_Y = 7
SQUARE_MM = 35.0   # nominal; measure the printed square and pass the real value
MARKER_MM = 26.0
DICT = cv2.aruco.DICT_5X5_100


def make_board(square_mm=SQUARE_MM, marker_mm=MARKER_MM):
    dictionary = cv2.aruco.getPredefinedDictionary(DICT)
    # lengths in metres so every downstream pose is metric
    return cv2.aruco.CharucoBoard(
        (SQUARES_X, SQUARES_Y), square_mm / 1000.0, marker_mm / 1000.0, dictionary
    )
