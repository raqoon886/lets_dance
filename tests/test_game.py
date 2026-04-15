"""Tests for game engine and session management."""

import pytest


class TestGameEngine:
    """Test suite for GameEngine."""

    def test_initial_state(self):
        """Test engine starts in MENU state."""
        from game.engine import GameEngine, GameState
        engine = GameEngine(config={})
        assert engine.state == GameState.MENU
        assert not engine.running

    def test_state_transition(self):
        """Test state transitions."""
        from game.engine import GameEngine, GameState
        engine = GameEngine(config={})
        engine.transition_to(GameState.SONG_SELECT)
        assert engine.state == GameState.SONG_SELECT

    def test_shutdown(self):
        """Test clean shutdown."""
        from game.engine import GameEngine
        engine = GameEngine(config={})
        engine.running = True
        engine.shutdown()
        assert not engine.running


class TestGameSession:
    """Test suite for GameSession."""

    def test_session_creation(self):
        """Test session initialization."""
        from game.session import GameSession
        session = GameSession("test_song", {"duration": 30})
        assert session.song_id == "test_song"
        assert not session.is_active

    def test_session_start(self):
        """Test session start sets active flag."""
        from game.session import GameSession
        session = GameSession("test_song", {"duration": 30})
        session.start()
        assert session.is_active
        assert session.start_time is not None

    def test_session_progress(self):
        """Test progress calculation."""
        from game.session import GameSession
        session = GameSession("test_song", {"duration": 30})
        assert session.progress == 0.0


class TestGameModes:
    """Test suite for game modes."""

    def test_practice_mode(self):
        """Test practice mode shows reference."""
        from game.modes import PracticeMode
        mode = PracticeMode()
        assert mode.should_show_reference()
        assert mode.get_time_limit() == 0

    def test_challenge_mode(self):
        """Test challenge mode has combo scoring."""
        from game.modes import ChallengeMode
        mode = ChallengeMode()
        config = mode.get_scoring_config()
        assert config["combo_enabled"]
        assert not mode.should_show_reference()

    def test_freestyle_mode(self):
        """Test freestyle mode has no scoring."""
        from game.modes import FreestyleMode
        mode = FreestyleMode()
        config = mode.get_scoring_config()
        assert not config["enabled"]
