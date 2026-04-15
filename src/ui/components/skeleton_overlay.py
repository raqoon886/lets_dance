"""
Skeleton Overlay Widget - Draws pose skeleton over the camera feed.
Supports user skeleton, reference ghost, silhouette mode, and comparison mode.
"""

import numpy as np


class SkeletonOverlayWidget:
    """
    Renders skeleton landmarks and connections over a camera feed surface.
    Supports drawing user pose and semi-transparent reference pose.
    Now also supports silhouette rendering via SilhouetteRenderer.
    """

    # Color presets
    USER_COLOR = (0, 255, 200)        # Cyan-green
    REFERENCE_COLOR = (255, 100, 255)  # Magenta
    MATCH_COLOR = (0, 255, 0)          # Green (matching well)
    MISMATCH_COLOR = (255, 50, 50)     # Red (mismatching)

    def __init__(self, frame_size: tuple, line_width=3, joint_radius=5,
                 use_silhouette: bool = True):
        """
        Args:
            frame_size: (width, height) of the camera frame
            line_width: Thickness of skeleton lines
            joint_radius: Radius of joint circles
            use_silhouette: If True, use SilhouetteRenderer for body shape
        """
        self.frame_size = frame_size
        self.line_width = line_width
        self.joint_radius = joint_radius
        self.use_silhouette = use_silhouette
        self._user_renderer = None
        self._ref_renderer = None

        if use_silhouette:
            from pose.silhouette_renderer import SilhouetteRenderer
            self._user_renderer = SilhouetteRenderer()
            self._ref_renderer = SilhouetteRenderer(
                palette={
                    "head":      (255, 140, 200),
                    "torso":     (255, 100, 180),
                    "left_arm":  (255, 130, 200),
                    "right_arm": (255, 130, 200),
                    "left_leg":  (255, 110, 180),
                    "right_leg": (255, 110, 180),
                    "joint":     (255, 200, 230),
                    "outline":   (100, 30, 60),
                },
            )

    def draw_user_skeleton(self, surface, landmarks: np.ndarray,
                            connections: list, offset: tuple = (0, 0)):
        """
        Draw the user's detected skeleton.

        Args:
            surface: PyGame surface to draw on
            landmarks: Joint positions — (33, 4) with visibility
            connections: List of (src, dst) joint index pairs
            offset: (x, y) offset for camera feed position
        """
        import pygame

        if self.use_silhouette and self._user_renderer is not None:
            w, h = self.frame_size
            panel = (offset[0], offset[1], w, h)
            self._user_renderer.draw(surface, landmarks, panel)
            return

        # Fallback: line-based skeleton
        for src, dst in connections:
            if src < len(landmarks) and dst < len(landmarks):
                p1 = (int(landmarks[src][0] * self.frame_size[0]) + offset[0],
                      int(landmarks[src][1] * self.frame_size[1]) + offset[1])
                p2 = (int(landmarks[dst][0] * self.frame_size[0]) + offset[0],
                      int(landmarks[dst][1] * self.frame_size[1]) + offset[1])
                pygame.draw.line(surface, self.USER_COLOR, p1, p2, self.line_width)

        for i in range(len(landmarks)):
            pt = (int(landmarks[i][0] * self.frame_size[0]) + offset[0],
                  int(landmarks[i][1] * self.frame_size[1]) + offset[1])
            pygame.draw.circle(surface, self.USER_COLOR, pt, self.joint_radius)

    def draw_reference_skeleton(self, surface, landmarks: np.ndarray,
                                 connections: list, opacity: int = 128,
                                 offset: tuple = (0, 0)):
        """
        Draw the reference skeleton as a semi-transparent ghost overlay.

        Args:
            surface: PyGame surface to draw on
            landmarks: Reference joint positions (33, 4)
            connections: Connection pairs
            opacity: Alpha value (0-255)
            offset: Position offset
        """
        if self.use_silhouette and self._ref_renderer is not None:
            w, h = self.frame_size
            panel = (offset[0], offset[1], w, h)
            self._ref_renderer.draw(surface, landmarks, panel, alpha=opacity)
            return

        # Fallback: line-based reference
        import pygame
        tmp = pygame.Surface(self.frame_size, pygame.SRCALPHA)
        for src, dst in connections:
            if src < len(landmarks) and dst < len(landmarks):
                p1 = (int(landmarks[src][0] * self.frame_size[0]),
                      int(landmarks[src][1] * self.frame_size[1]))
                p2 = (int(landmarks[dst][0] * self.frame_size[0]),
                      int(landmarks[dst][1] * self.frame_size[1]))
                pygame.draw.line(tmp, (*self.REFERENCE_COLOR, opacity),
                                 p1, p2, self.line_width)
        surface.blit(tmp, offset)

    def draw_comparison(self, surface, user_landmarks: np.ndarray,
                        ref_landmarks: np.ndarray, connections: list,
                        threshold: float = 0.1, offset: tuple = (0, 0)):
        """
        Draw user skeleton color-coded by match quality with reference.
        Green = close match, Red = mismatch.

        Args:
            surface: PyGame surface to draw on
            user_landmarks: User joint positions (33, 4)
            ref_landmarks: Reference joint positions (33, 4)
            connections: Connection pairs
            threshold: Distance threshold for color coding
            offset: Position offset
        """
        import pygame

        for src, dst in connections:
            if (src < len(user_landmarks) and dst < len(user_landmarks) and
                    src < len(ref_landmarks) and dst < len(ref_landmarks)):
                # Per-joint distance
                d1 = np.linalg.norm(user_landmarks[src][:2] - ref_landmarks[src][:2])
                d2 = np.linalg.norm(user_landmarks[dst][:2] - ref_landmarks[dst][:2])
                avg_d = (d1 + d2) / 2.0

                # Interpolate color
                t = min(avg_d / threshold, 1.0)
                color = tuple(
                    int(self.MATCH_COLOR[c] * (1 - t) + self.MISMATCH_COLOR[c] * t)
                    for c in range(3)
                )

                p1 = (int(user_landmarks[src][0] * self.frame_size[0]) + offset[0],
                      int(user_landmarks[src][1] * self.frame_size[1]) + offset[1])
                p2 = (int(user_landmarks[dst][0] * self.frame_size[0]) + offset[0],
                      int(user_landmarks[dst][1] * self.frame_size[1]) + offset[1])
                pygame.draw.line(surface, color, p1, p2, self.line_width)
