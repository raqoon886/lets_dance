# Let's Dance - On-Device AI Dance Scoring Game

An on-device AI dance game for Raspberry Pi that uses a webcam to detect poses via MediaPipe, extracts motion embeddings through a Spatio-Temporal Graph Convolutional Network (ST-GCN), and scores dance similarity in real-time.

## Architecture Overview

```
Webcam → MediaPipe Pose → Landmark Normalization → Skeleton Graph → ST-GCN Embedding → Cosine Similarity → Score
                                                                        ↑
                                                          Reference Embeddings (pre-recorded)
```

## Project Structure

```
lets_dance/
├── config/
│   └── settings.yaml              # Game configuration (camera, model, UI, scoring)
├── src/
│   ├── main.py                    # Application entry point
│   ├── pose/                      # Pose estimation module
│   │   ├── detector.py            # MediaPipe pose detection wrapper
│   │   ├── landmark_utils.py      # Landmark normalization, smoothing, angles
│   │   └── visualizer.py          # Skeleton overlay rendering
│   ├── embedding/                 # Dance embedding module
│   │   ├── graph_builder.py       # Skeleton → graph data structure
│   │   ├── model.py               # ST-GCN embedding model (PyTorch Geometric)
│   │   └── extractor.py           # End-to-end embedding extraction pipeline
│   ├── scoring/                   # Scoring module
│   │   ├── similarity.py          # Cosine / Euclidean / DTW similarity
│   │   ├── scorer.py              # Score calculation, combos, grades
│   │   └── feedback.py            # Real-time feedback generation
│   ├── game/                      # Game engine module
│   │   ├── engine.py              # Main game loop and state machine
│   │   ├── session.py             # Single dance play session
│   │   └── modes.py               # Practice / Challenge / Freestyle modes
│   └── ui/                        # UI module (PyGame)
│       ├── app.py                 # Main UI application
│       ├── screens/               # Full-screen views
│       │   ├── home.py            # Main menu
│       │   ├── song_select.py     # Song/dance browser
│       │   ├── gameplay.py        # Main dance gameplay view
│       │   ├── result.py          # Score results screen
│       │   └── settings.py        # Settings panel
│       └── components/            # Reusable UI widgets
│           ├── camera_feed.py     # Live webcam display
│           ├── score_display.py   # Animated score counter
│           ├── progress_bar.py    # Song progress bar
│           ├── skeleton_overlay.py# Skeleton visualization
│           └── countdown.py       # 3-2-1-GO countdown
├── assets/                        # Static assets
│   ├── fonts/                     # TTF font files
│   ├── images/                    # Logo, backgrounds, icons
│   └── sounds/                    # Sound effects (perfect.wav, etc.)
├── data/
│   ├── reference_dances/          # Pre-recorded reference dance data
│   └── models/                    # Trained model weights (.pth)
├── scripts/
│   ├── collect_reference.py       # Record reference dance data
│   ├── train_embedding.py         # Train the ST-GCN embedding model
│   └── calibrate_camera.py        # Camera setup & calibration
├── tests/                         # Unit tests
│   ├── test_pose.py
│   ├── test_embedding.py
│   ├── test_scoring.py
│   └── test_game.py
├── requirements.txt
├── setup.py
└── .gitignore
```

## Game Flow (UI/UX)

```
┌──────────┐    ┌─────────────┐    ┌───────────┐    ┌─────────────┐    ┌──────────┐
│   Home   │───>│ Song Select │───>│ Countdown │───>│  Gameplay   │───>│  Result  │
│   Menu   │    │   Screen    │    │  3-2-1-GO │    │   Screen    │    │  Screen  │
└──────────┘    └─────────────┘    └───────────┘    └─────────────┘    └──────────┘
     │                                                                       │
     │<──────────────────────────────────────────────────────────────────────┘
     │                                                                  (Retry / Menu)
     v
┌──────────┐
│ Settings │
└──────────┘
```

### Game Modes

| Mode | Reference Overlay | Scoring | Combo | Description |
|------|:-:|:-:|:-:|---|
| **Practice** | Yes | Lenient | No | Learn at your own pace |
| **Challenge** | No | Strict | Yes | Score as high as you can |
| **Freestyle** | No | Off | No | Free dance + motion capture |

### Scoring System

| Grade | Threshold | Feedback |
|-------|-----------|----------|
| Perfect | ≥ 90% | Gold glow + particles |
| Great | ≥ 75% | Green flash |
| Good | ≥ 60% | Cyan indicator |
| OK | ≥ 40% | Gray text |
| Miss | < 40% | Red flash |

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Calibrate camera
python scripts/calibrate_camera.py --camera 0

# 3. Record a reference dance
python scripts/collect_reference.py --song "my_dance" --duration 60

# 4. Train embedding model
python scripts/train_embedding.py --data-dir data/reference_dances --epochs 100

# 5. Run the game
python src/main.py
```

## Hardware Requirements

- Raspberry Pi 4 (4GB+ recommended)
- USB Webcam (720p+)
- Display (HDMI, 1024x600+ recommended)
- Speaker (for music & sound effects)

## Tech Stack

- **Pose Estimation**: MediaPipe Pose (33 landmarks)
- **Embedding Model**: ST-GCN (Spatio-Temporal Graph Convolutional Network)
- **ML Framework**: PyTorch + PyTorch Geometric
- **Game UI**: PyGame
- **Camera**: OpenCV
- **Config**: YAML
