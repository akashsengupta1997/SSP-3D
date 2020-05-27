import numpy as np


def convert_bbox_corners_to_centre_hw(bbox_corners):
    """
    Converst bbox coordinates from x1, y1, x2, y2 to centre, height, width.
    """
    x1, y1, x2, y2 = bbox_corners
    centre = np.array([(x1+x2)/2.0, (y1+y2)/2.0])
    height = x2 - x1
    width = y2 - y1

    return centre, height, width


def convert_bbox_centre_hw_to_corners(centre, height, width):
    x1 = centre[0] - height/2.0
    x2 = centre[0] + height/2.0
    y1 = centre[1] - width/2.0
    y2 = centre[1] + width/2.0

    return np.array([x1, y1, x2, y2], dtype=np.int16)


