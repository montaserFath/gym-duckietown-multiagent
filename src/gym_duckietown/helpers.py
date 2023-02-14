"""
Helpers functions to
- detect lines from an image
- get line in camera's coordinates
- get line in bot's coordinates
"""
import numpy as np


def detect_lines_from_img(img: np.ndarray) -> dict:
    """
    detect yellow, red and white line from an image and return lines location in pixels
    return dict
    """
    lines = {"yellow": [], "red": [], "white": []}

    return lines


def pixel_to_bot(lines: dict, inter_param: np.ndarray, exter_param: np.ndarray) -> dict:
    """

    """
    lines_bot = {}

    return lines_bot
