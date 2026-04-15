"""
Camera Feed Widget - Displays the live webcam feed in the game UI.
Handles frame conversion from OpenCV to PyGame surface.
"""

import numpy as np


class CameraFeedWidget:
    """Renders the live camera feed as a PyGame surface."""

    def __init__(self, rect: tuple):
        """
        Args:
            rect: (x, y, width, height) display region
        """
        self.rect = rect
        self._surface = None

    def update(self, frame: np.ndarray):
        """
        Convert an OpenCV BGR frame to a PyGame surface.

        Args:
            frame: BGR image from webcam (H, W, 3)
        """
        # TODO:
        # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # frame_resized = cv2.resize(frame_rgb, (self.rect[2], self.rect[3]))
        # self._surface = pygame.surfarray.make_surface(
        #     frame_resized.swapaxes(0, 1)
        # )
        pass

    def render(self, display):
        """Draw the camera feed onto the display surface."""
        # TODO:
        # if self._surface:
        #     display.blit(self._surface, (self.rect[0], self.rect[1]))
        # else:
        #     # Draw placeholder "No Camera" message
        #     pygame.draw.rect(display, (30, 30, 30), self.rect)
        pass
