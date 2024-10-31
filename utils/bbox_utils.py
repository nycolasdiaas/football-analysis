from typing import List, Tuple
from Enums import Position
import numpy as np

def get_center_of_bbox(bbox):
    x1,y1,x2,y2 = bbox
    return int( (x1+x2)/2 ), int( (y1+y2)/2 )

def get_bbox_width(bbox):
    return bbox[2]-bbox[0]

def measure_distance(p1,p2):
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5

def get_anchors_coordinates(bbox: List[int],anchor: Position) -> np.ndarray:
    """
    Calculates and returns the coordinates of a specific anchor point
    within the bounding boxes defined by the `xyxy` attribute. The anchor
    point can be any of the predefined positions in the `Position` enum,
    such as `CENTER`, `CENTER_LEFT`, `BOTTOM_RIGHT`, etc.

    Args:
        anchor (Position): An enum specifying the position of the anchor point
            within the bounding box. Supported positions are defined in the
            `Position` enum.

    Returns:
        np.ndarray: An array of shape `(n, 2)`, where `n` is the number of bounding
            boxes. Each row contains the `[x, y]` coordinates of the specified
            anchor point for the corresponding bounding box.

    Raises:
        ValueError: If the provided `anchor` is not supported.
    """
    if anchor == Position.CENTER:
        return np.array(
            [
                (bbox[0] + bbox[2]) / 2,
                (bbox[1] + bbox[3]) / 2,
            ]
        ).transpose()
    elif anchor == Position.CENTER_LEFT:
        return np.array(
            [
                bbox[0],
                (bbox[1] + bbox[3]) / 2,
            ]
        ).transpose()
    elif anchor == Position.CENTER_RIGHT:
        return np.array(
            [
                bbox[2],
                (bbox[1] + bbox[3]) / 2,
            ]
        ).transpose()
    elif anchor == Position.BOTTOM_CENTER:
        return np.array(
            [(bbox[0] + bbox[2]) / 2, bbox[3]]
        ).transpose()
    elif anchor == Position.BOTTOM_LEFT:
        return np.array([bbox[0], bbox[3]]).transpose()
    elif anchor == Position.BOTTOM_RIGHT:
        return np.array([bbox[2], bbox[3]]).transpose()
    elif anchor == Position.TOP_CENTER:
        return np.array(
            [(bbox[0] + bbox[2]) / 2, bbox[1]]
        ).transpose()
    elif anchor == Position.TOP_LEFT:
        return np.array([bbox[0], bbox[1]]).transpose()
    elif anchor == Position.TOP_RIGHT:
        return np.array([bbox[2], bbox[1]]).transpose()

    raise ValueError(f"{anchor} is not supported.")

def point_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """
    Calculate the Euclidean distance between two points.

    Args:
        p1 (Tuple[float, float]): The first point (x1, y1).
        p2 (Tuple[float, float]): The second point (x2, y2).

    Returns:
        float: The distance between the two points.
    """
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def xy_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> Tuple[float, float]:
    """
    Calculate the horizontal and vertical distances between two points.

    Args:
        p1 (Tuple[float, float]): The first point (x1, y1).
        p2 (Tuple[float, float]): The second point (x2, y2).

    Returns:
        Tuple[float, float]: The horizontal and vertical distances between the two points.
    """
    return p1[0] - p2[0], p1[1] - p2[1]