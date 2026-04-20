"""
Main UI Application - PyGame-based game interface.
Manages screen routing, asset loading, and display lifecycle.
"""

import os
import yaml


class App:
    """
    Top-level UI application. Initializes PyGame display, manages
    screen transitions, and provides shared resources to screen objects.
    """

    def __init__(self, width=1024, height=600, fullscreen=False, theme="neon"):
        self.width = width
        self.height = height
        self.fullscreen = fullscreen
        self.theme = theme
        self._display = None
        self._fonts = {}
        self._images = {}
        self._sounds = {}
        self._screens = {}
        self._active_screen = None

    def initialize(self):
        """
        Initialize PyGame display and load all assets.
        """
        # TODO:
        # pygame.init()
        # flags = pygame.FULLSCREEN if self.fullscreen else 0
        # self._display = pygame.display.set_mode((self.width, self.height), flags)
        # pygame.display.set_caption("Let's Dance!")
        # self._load_assets()
        # self._init_screens()
        pass

    def _load_assets(self):
        """Load fonts, images, and sound effects."""
        # TODO:
        # self._fonts["title"] = pygame.font.Font("assets/fonts/title.ttf", 48)
        # self._fonts["score"] = pygame.font.Font("assets/fonts/score.ttf", 36)
        # self._fonts["body"] = pygame.font.Font(None, 24)
        # self._images["logo"] = pygame.image.load("assets/images/logo.png")
        # self._images["bg"] = pygame.image.load("assets/images/background.png")

        # Read sfx_volume from config
        sfx_volume = 0.8
        try:
            with open("config/settings.yaml", "r") as f:
                cfg = yaml.safe_load(f) or {}
            sfx_volume = float(cfg.get("audio", {}).get("sfx_volume", sfx_volume))
        except Exception:
            pass

        # Load SFX
        import pygame
        sfx_path = os.path.join("assets", "sounds", "click.mp3")
        if os.path.exists(sfx_path):
            click_sound = pygame.mixer.Sound(sfx_path)
            click_sound.set_volume(sfx_volume)
            self._sounds["click"] = click_sound

    def play_sfx(self, name: str):
        """Play a sound effect by name. Volume follows audio.sfx_volume in settings.yaml."""
        sound = self._sounds.get(name)
        if sound:
            sound.play()

    def _init_screens(self):
        """Initialize all screen objects."""
        # TODO: Import and instantiate each screen
        # from .screens import HomeScreen, SongSelectScreen, GameplayScreen, ...
        pass

    def set_screen(self, screen_name: str):
        """Switch to a different screen."""
        # TODO: Transition animation, set active screen
        self._active_screen = self._screens.get(screen_name)

    def render(self, game_state: dict):
        """
        Render the current screen.

        Args:
            game_state: Current game state data to display
        """
        # TODO: Clear display, render active screen, flip
        # self._display.fill((0, 0, 0))
        # if self._active_screen:
        #     self._active_screen.render(self._display, game_state)
        # pygame.display.flip()
        pass

    def get_display(self):
        """Return the pygame display surface."""
        return self._display

    def shutdown(self):
        """Clean up PyGame resources."""
        # TODO: pygame.quit()
        pass
