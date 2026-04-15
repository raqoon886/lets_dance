"""
Skeleton Overlay Widget - Draws pose skeleton over the camera feed.
Supports user skeleton, reference ghost, and comparison mode.
"""

import numpy as np


class SkeletonOverlayWidget:
    """
    Renders skeleton landmarks and connections over a camera feed surface.
    Supports drawing user pose and semi-transparent reference pose.
    """

    # Color presets
    USER_COLOR = (0, 255, 200)        # Cyan-green
    REFERENCE_COLOR = (255, 100, 255)  # Magenta
    MATCH_COLOR = (0, 255, 0)          # Green (matching well)
    MISMATCH_COLOR = (255, 50, 50)     # Red (mismatching)

    def __init__(self, frame_size: tuple, line_width=3, joint_radius=5):
        """
        Args:
            frame_size: (width, height) of the camera frame
            line_width: Thickness of skeleton lines
            joint_radius: Radius of joint circles
        """
        self.frame_size = frame_size
        self.line_width = line_width
        self.joint_radius = joint_radius

    def draw_user_skeleton(self, surface, landmarks: np.ndarray,
                            connections: list, offset: tuple = (0, 0)):
        """
        Draw the user's detected skeleton.

        Args:
            surface: PyGame surface to draw on
            landmarks: Joint positions in pixel coords (N, 2)
            connections: List of (src, dst) joint index pairs
            offset: (x, y) offset for camera feed position
        """
        # TODO: Draw connections as lines, joints as circles
        # Color-code by body part (arms, legs, torso)
        pass

    def draw_reference_skeleton(self, surface, landmarks: np.ndarray,
                                 connections: list, opacity: int = 128,
                                 offset: tuple = (0, 0)):
        """
        Draw the reference skeleton as a semi-transparent ghost overlay.

        Args:
            surface: PyGame surface to draw on
            landmarks: Reference joint positions (N, 2)
            connections: Connection pairs
            opacity: Alpha value (0-255)
            offset: Position offset
        """
        # TODO: Draw on a temporary surface with alpha, then blit
        pass

    def draw_comparison(self, surface, user_landmarks: np.ndarray,
                        ref_landmarks: np.ndarray, connections: list,
                        threshold: float = 0.1, offset: tuple = (0, 0)):
        """
        Draw user skeleton color-coded by match quality with reference.
        Green = close match, Red = mismatch.

        Args:
            surface: PyGame surface to draw on
            user_landmarks: User joint positions (N, 2)
            ref_landmarks: Reference joint positions (N, 2)
            connections: Connection pairs
            threshold: Distance threshold for color coding
            offset: Position offset
        """
        # TODO: Compute per-joint distance, interpolate color between
        # MATCH_COLOR and MISMATCH_COLOR based on distance
        pass
