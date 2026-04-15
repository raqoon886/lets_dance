"""
Pose Visualizer - Draws skeleton overlay on camera frames.
Supports both simple skeleton lines and silhouette rendering.
"""

import numpy as np


class PoseVisualizer:
    """Renders pose landmarks and skeleton connections onto frames."""

    # Color scheme (BGR) for different body parts
    COLORS = {
        "left_arm": (0, 255, 128),
        "right_arm": (128, 255, 0),
        "left_leg": (255, 128, 0),
        "right_leg": (0, 128, 255),
        "torso": (255, 255, 0),
        "joint": (0, 255, 255),
        "reference": (255, 0, 255),
    }

    def __init__(self, line_thickness=2, joint_radius=4):
        self.line_thickness = line_thickness
        self.joint_radius = joint_radius

    def draw_skeleton(self, frame: np.ndarray, landmarks: np.ndarray,
                      connections: list, color_map=None) -> np.ndarray:
        """
        Draw skeleton overlay on the frame.

        Args:
            frame: BGR image (H, W, 3)
            landmarks: Landmark positions in pixel coords (N, 2)
            connections: List of (src, dst) joint index pairs
            color_map: Optional dict mapping connection indices to colors

        Returns:
            Frame with skeleton drawn on it
        """
        import cv2
        output = frame.copy()
        h, w = output.shape[:2]
        default_color = self.COLORS["torso"]

        # Draw connections
        for i, (src, dst) in enumerate(connections):
            if src < len(landmarks) and dst < len(landmarks):
                pt1 = (int(landmarks[src][0] * w), int(landmarks[src][1] * h))
                pt2 = (int(landmarks[dst][0] * w), int(landmarks[dst][1] * h))
                color = color_map.get(i, default_color) if color_map else default_color
                cv2.line(output, pt1, pt2, color, self.line_thickness)

        # Draw joints
        for i in range(len(landmarks)):
            pt = (int(landmarks[i][0] * w), int(landmarks[i][1] * h))
            cv2.circle(output, pt, self.joint_radius, self.COLORS["joint"], -1)

        return output

    def draw_comparison(self, frame: np.ndarray,
                        user_landmarks: np.ndarray,
                        reference_landmarks: np.ndarray,
                        connections: list) -> np.ndarray:
        """
        Draw both user and reference skeletons for visual comparison.
        """
        import cv2
        output = frame.copy()
        h, w = output.shape[:2]

        # Draw reference skeleton (semi-transparent magenta)
        overlay = output.copy()
        for src, dst in connections:
            if src < len(reference_landmarks) and dst < len(reference_landmarks):
                pt1 = (int(reference_landmarks[src][0] * w), int(reference_landmarks[src][1] * h))
                pt2 = (int(reference_landmarks[dst][0] * w), int(reference_landmarks[dst][1] * h))
                cv2.line(overlay, pt1, pt2, self.COLORS["reference"], self.line_thickness)
        cv2.addWeighted(overlay, 0.4, output, 0.6, 0, output)

        # Draw user skeleton (solid)
        output = self.draw_skeleton(output, user_landmarks, connections)
        return output

    def draw_feedback_indicator(self, frame: np.ndarray,
                                 score: float, position: tuple) -> np.ndarray:
        """
        Draw real-time score feedback indicator (Perfect/Great/Good/Miss).
        """
        import cv2
        output = frame.copy()
        if score >= 90:
            text, color = "PERFECT!", (0, 215, 255)
        elif score >= 75:
            text, color = "GREAT!", (128, 255, 0)
        elif score >= 60:
            text, color = "GOOD", (255, 200, 0)
        elif score >= 40:
            text, color = "OK", (200, 200, 200)
        else:
            text, color = "MISS", (50, 50, 255)

        cv2.putText(output, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                     1.2, color, 3, cv2.LINE_AA)
        return output
