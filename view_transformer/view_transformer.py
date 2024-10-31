from typing import Any, Dict, List, Tuple, Optional

import cv2
import pandas as pd
from .base import BaseTransformer
from .homography import Homography
from utils import get_anchors_coordinates
from Enums import Position
from supervision import KeyPoints

import numpy as np

class ViewTransformer(BaseTransformer):
    """
    A class to map object positions from detected keypoints to a top-down view.

    This class implements the mapping of detected objects to their corresponding
    positions in a top-down representation based on the homography obtained from 
    detected keypoints.
    """

    def __init__(self, top_down_keypoints: np.ndarray, alpha: float = 0.9) -> None:
        """
        Initializes the ObjectPositionMapper.

        Args:
            top_down_keypoints (np.ndarray): An array of shape (n, 2) containing the top-down keypoints.
            alpha (float): Smoothing factor for homography smoothing.
        """
        super().__init__()
        self.top_down_keypoints = top_down_keypoints
        self.homography: Homography = Homography(alpha)
    
    def transform(self, object_tracks: List[List[Dict[str, Any]]], keypoints_tracks: Optional[KeyPoints], filter: np.ndarray) -> List[List[Dict[str, Any]]]:
        """Maps the detection data to their positions in the top-down view.

        This method retrieves keypoints and object information from the detection data,
        computes the homography matrix, smooths it over frames, and projects the foot positions
        of detected objects.

        Args:
            object_tracks (List[List[Dict[str, Any]]]): Object tracks containing the object information.
            keypoints_tracks (Optional[KeyPoints]): Detected keypoints. May be None if not available.
            filter (np.ndarray[bool]): A boolean array to filter the keypoints.
        
        Returns:
            List[List[Dict[str, Any]]]: The object tracks with projected positions.
        """
        
        # Se keypoints_tracks for None ou vazio, defina keypoints como um array vazio
        if keypoints_tracks is None or len(keypoints_tracks) == 0:
            keypoints = np.empty((0, 2))  # Um array vazio com 0 linhas e 2 colunas
        else:
            keypoints = keypoints_tracks.xy[0]

        # Matriz de homografia identidade se não houver keypoints
        if keypoints.size == 0 or keypoints.shape[1] != 2:
            homography_matrix = np.eye(3)
        else:
            homography_matrix = self.homography.find_homography(keypoints, self.top_down_keypoints[filter])

        transformed_tracks = []
        for player_track in object_tracks:
            transformed_player_track = []
            for track_info in player_track:
                if isinstance(track_info, dict) and 'bbox' in track_info:
                    bbox = track_info['bbox']
                    feet_pos = get_anchors_coordinates(bbox, anchor=Position.BOTTOM_CENTER)
                    projected_pos = self.homography.perspective_transform(posistion=feet_pos, H_mat=homography_matrix)
                    track_info['projection'] = projected_pos
                    track_info['position'] = feet_pos
                    transformed_player_track.append(track_info)
                else:
                    print("Track não é um dicionário ou não tem 'bbox':", track_info)
            transformed_tracks.append(transformed_player_track)

        return transformed_tracks
    
    def adjust_transforms(self, object_tracks: List[List[Dict[str, Any]]], camera_movement: Tuple[float, float]) -> List[List[Dict[str, Any]]]:
        """
        Adjust the projected positions of the objects based on the camera movement in each frame.

        Args:
            object_tracks (List[List[Dict[str, Any]]]): Object tracks containing the object information.
            camera_movement (Tuple[float, float]): The camera movement in frame.

        Returns:
            List[List[Dict[str, Any]]]: The object tracks with adjusted projected positions.
        """
        for player_track in object_tracks:
            for track_info in player_track:
                if 'projection' in track_info:
                    position = track_info['projection']
                    position_adjusted = (position[0] - camera_movement[0], position[1] - camera_movement[1])
                    track_info['projection'] = position_adjusted

        return object_tracks
