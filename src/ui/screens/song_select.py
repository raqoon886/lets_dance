"""
Song Select Screen - Browse and choose a dance/song.

Layout:
┌─────────────────────────────────────┐
│  ← Back         SELECT A SONG       │
│ ┌──────┐ ┌──────┐ ┌──────┐         │
│ │ Song │ │ Song │ │ Song │  ...     │
│ │  1   │ │  2   │ │  3   │         │
│ │ ★★★  │ │ ★★☆  │ │ ★☆☆  │         │
│ └──────┘ └──────┘ └──────┘         │
│                                     │
│  Preview: [thumbnail]  Duration: 2m │
│  Difficulty: ★★★   BPM: 120        │
│              [ ▶ START ]            │
└─────────────────────────────────────┘
"""

import os


class SongSelectScreen:
    """Song/dance selection screen with previews and difficulty info."""

    def __init__(self, app):
        self.app = app
        self._songs = []
        self._selected_index = 0
        self._scroll_offset = 0

    def load_songs(self, data_dir: str = "data/reference_dances"):
        """
        Scan reference dance directory and load available songs.

        Args:
            data_dir: Path to reference dance data
        """
        # TODO: Scan data_dir for song folders, load metadata.json from each
        # Expected structure:
        #   data/reference_dances/song_name/
        #       metadata.json    (title, artist, duration, bpm, difficulty)
        #       reference.npy    (pre-recorded landmark/embedding data)
        #       thumbnail.png    (preview image)
        self._songs = [
            {
                "id": "demo_song",
                "title": "Demo Dance",
                "artist": "Let's Dance",
                "duration": 60,
                "bpm": 120,
                "difficulty": 2,
                "path": os.path.join(data_dir, "demo_song"),
            },
        ]

    def handle_input(self, events: list) -> dict:
        """
        Handle navigation and selection.

        Returns:
            {'action': 'start', 'song': song_data} or
            {'action': 'back'} or None
        """
        # TODO: LEFT/RIGHT to browse, ENTER to select, ESC to go back
        return None

    def render(self, display, game_state: dict):
        """
        Render song selection grid with preview panel.

        Args:
            display: PyGame display surface
            game_state: Current state data
        """
        # TODO:
        # 1. Draw header with back button
        # 2. Draw scrollable song card grid
        # 3. Draw selected song preview panel (thumbnail, info, start button)
        # 4. Draw difficulty stars and BPM info
        pass

    @property
    def selected_song(self) -> dict:
        """Return currently selected song data."""
        if self._songs and 0 <= self._selected_index < len(self._songs):
            return self._songs[self._selected_index]
        return None
