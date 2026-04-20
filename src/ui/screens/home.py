"""
Home Screen - Main menu with mode selection and navigation.

Layout:
┌─────────────────────────────────────┐
│           🎵 LET'S DANCE 🎵        │
│                                     │
│         [ ▶ PRACTICE MODE ]         │
│         [ 🏆 CHALLENGE MODE ]       │
│         [ 🎤 FREESTYLE MODE ]       │
│                                     │
│     [ ⚙ Settings ]  [ ❓ Help ]    │
└─────────────────────────────────────┘
"""


class HomeScreen:
    """Main menu screen with animated background and mode selection buttons."""

    def __init__(self, app):
        self.app = app
        self._buttons = []
        self._animation_frame = 0
        self._selected_index = 0

    def setup(self):
        """Initialize menu buttons and animations."""
        # TODO: Create button objects for each game mode
        # Buttons: Practice, Challenge, Freestyle, Settings, Help
        self._buttons = [
            {"label": "PRACTICE MODE", "action": "practice", "icon": "play"},
            {"label": "CHALLENGE MODE", "action": "challenge", "icon": "trophy"},
            {"label": "FREESTYLE MODE", "action": "freestyle", "icon": "mic"},
        ]

    def handle_input(self, events: list) -> str:
        """
        Process input events and return action string if button pressed.

        Args:
            events: List of pygame events

        Returns:
            Action string (e.g., 'practice', 'settings') or None
        """
        import pygame

        if not self._buttons:
            return None

        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    self._selected_index = (self._selected_index - 1) % len(self._buttons)
                elif event.key == pygame.K_DOWN:
                    self._selected_index = (self._selected_index + 1) % len(self._buttons)
                elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                    self.app.play_sfx("click")
                    return self._buttons[self._selected_index]["action"]
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                # TODO: check event.pos against each button's rect when rects are defined
                self.app.play_sfx("click")

        return None

    def render(self, display, game_state: dict):
        """
        Render the home screen.

        Args:
            display: PyGame display surface
            game_state: Current state data
        """
        # TODO:
        # 1. Draw animated gradient background
        # 2. Draw logo / title text with glow effect
        # 3. Draw menu buttons with hover/selection highlight
        # 4. Draw footer with settings and help icons
        # 5. Animate idle camera preview in background
        pass

    def _draw_title(self, display):
        """Draw the game title with animation."""
        # TODO: Pulsating neon text effect
        pass

    def _draw_buttons(self, display):
        """Draw mode selection buttons."""
        # TODO: Rounded rect buttons with icon + label, highlight selected
        pass
