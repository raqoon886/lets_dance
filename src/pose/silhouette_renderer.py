"""
Silhouette Renderer – 포즈 랜드마크 기반으로 사람 형태 실루엣을 그립니다.

단순한 점·선 스틱 피겨 대신, 머리(원), 몸통·팔·다리를 두꺼운 둥근 선과
타원 영역으로 표현해 사람 모양에 가까운 3차원적 캐릭터를 만듭니다.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

# MediaPipe 33 landmark indices
_NOSE = 0
_L_SHOULDER = 11
_R_SHOULDER = 12
_L_ELBOW = 13
_R_ELBOW = 14
_L_WRIST = 15
_R_WRIST = 16
_L_HIP = 23
_R_HIP = 24
_L_KNEE = 25
_R_KNEE = 26
_L_ANKLE = 27
_R_ANKLE = 28

# 부위별 연결 – (시작, 끝, 두께 비율, 색상 키)
_BODY_SEGMENTS = [
    # 몸통 (두꺼움)
    (_L_SHOULDER, _R_SHOULDER, 0.22, "torso"),
    (_L_SHOULDER, _L_HIP, 0.18, "torso"),
    (_R_SHOULDER, _R_HIP, 0.18, "torso"),
    (_L_HIP, _R_HIP, 0.20, "torso"),
    # 왼팔
    (_L_SHOULDER, _L_ELBOW, 0.12, "left_arm"),
    (_L_ELBOW, _L_WRIST, 0.10, "left_arm"),
    # 오른팔
    (_R_SHOULDER, _R_ELBOW, 0.12, "right_arm"),
    (_R_ELBOW, _R_WRIST, 0.10, "right_arm"),
    # 왼다리
    (_L_HIP, _L_KNEE, 0.14, "left_leg"),
    (_L_KNEE, _L_ANKLE, 0.11, "left_leg"),
    # 오른다리
    (_R_HIP, _R_KNEE, 0.14, "right_leg"),
    (_R_KNEE, _R_ANKLE, 0.11, "right_leg"),
]

# 관절 인덱스 목록 (원 형태로 관절을 강조)
_JOINT_INDICES = [
    _L_SHOULDER, _R_SHOULDER, _L_ELBOW, _R_ELBOW,
    _L_WRIST, _R_WRIST, _L_HIP, _R_HIP,
    _L_KNEE, _R_KNEE, _L_ANKLE, _R_ANKLE,
]

# 기본 색상 팔레트 (BGR 아닌 RGB)
DEFAULT_PALETTE = {
    "head":      (100, 220, 255),
    "torso":     (80, 180, 255),
    "left_arm":  (60, 200, 180),
    "right_arm": (60, 200, 180),
    "left_leg":  (100, 160, 255),
    "right_leg": (100, 160, 255),
    "joint":     (200, 240, 255),
    "outline":   (30, 60, 100),
}


class SilhouetteRenderer:
    """
    포즈 랜드마크(33×4)를 입력받아 사람 형태의 실루엣을 pygame Surface에 그립니다.

    특징:
     - 머리: 코·어깨 위치 기반 원
     - 몸통/팔/다리: 둥근 끝(round cap)이 있는 두꺼운 선분
     - 관절: 작은 밝은 원
     - 외곽선(outline): 약간 더 큰 어두운 선으로 테두리 효과
    """

    def __init__(
        self,
        palette: Optional[dict] = None,
        visibility_threshold: float = 0.3,
        outline: bool = True,
        glow: bool = True,
    ):
        self.palette = palette or dict(DEFAULT_PALETTE)
        self.vis_thr = visibility_threshold
        self.outline = outline
        self.glow = glow

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #
    def draw(
        self,
        surface,            # pygame.Surface
        landmarks: np.ndarray,  # (33, 4) – x, y, z, visibility
        panel_rect: Tuple[int, int, int, int],  # (px, py, pw, ph)
        *,
        palette_override: Optional[dict] = None,
        alpha: int = 255,
    ):
        """
        panel_rect 영역 안에 사람 형태 실루엣을 그립니다.

        Args:
            surface: pygame.Surface (화면)
            landmarks: (33, 4) numpy array
            panel_rect: (x, y, w, h) 그리기 대상 패널 영역
            palette_override: 색상 팔레트를 일시적으로 교체
            alpha: 전체 투명도 (0~255). 255 미만이면 별도 Surface로 블렌딩.
        """
        import pygame

        palette = palette_override or self.palette
        px, py, pw, ph = panel_rect

        # ── 랜드마크 → 패널 내 픽셀 좌표 변환 ────────────────────
        pts = self._landmarks_to_pixels(landmarks, panel_rect)
        if pts is None:
            return  # 유효 랜드마크 없음

        # ── 투명 Surface (alpha < 255) ───────────────────────────
        if alpha < 255:
            tmp = pygame.Surface((pw, ph), pygame.SRCALPHA)
            tmp.fill((0, 0, 0, 0))
            offset = (-px, -py)
        else:
            tmp = surface
            offset = (0, 0)

        # 캐릭터 크기 참조 (어깨 너비 기준)
        ls = pts[_L_SHOULDER]
        rs = pts[_R_SHOULDER]
        shoulder_width = max(math.dist(ls, rs), 1.0)
        base_thick = max(int(shoulder_width * 0.22), 4)

        # ── 1) 외곽선(outline) ───────────────────────────────────
        if self.outline:
            outline_color = palette.get("outline", (30, 60, 100))
            self._draw_body_segments(
                tmp, pts, landmarks, base_thick, palette,
                offset, width_add=4, color_override=outline_color,
            )
            # 머리 외곽선
            self._draw_head(
                tmp, pts, landmarks, shoulder_width, palette,
                offset, radius_add=3, color_override=outline_color,
            )

        # ── 2) 글로우 효과 ───────────────────────────────────────
        if self.glow:
            glow_surface = pygame.Surface((pw, ph), pygame.SRCALPHA)
            glow_surface.fill((0, 0, 0, 0))
            glow_offset = (-px, -py) if alpha == 255 else offset
            self._draw_body_segments(
                glow_surface, pts, landmarks, base_thick, palette,
                glow_offset, width_add=10, alpha_mult=0.15,
            )
            self._draw_head(
                glow_surface, pts, landmarks, shoulder_width, palette,
                glow_offset, radius_add=8, alpha_mult=0.15,
            )
            if alpha < 255:
                tmp.blit(glow_surface, (0, 0))
            else:
                surface.blit(glow_surface, (px, py))

        # ── 3) 메인 실루엣 ───────────────────────────────────────
        self._draw_body_segments(
            tmp, pts, landmarks, base_thick, palette, offset,
        )
        self._draw_head(
            tmp, pts, landmarks, shoulder_width, palette, offset,
        )

        # ── 4) 관절 하이라이트 ───────────────────────────────────
        joint_color = palette.get("joint", (200, 240, 255))
        joint_r = max(int(base_thick * 0.35), 3)
        for idx in _JOINT_INDICES:
            if landmarks[idx][3] < self.vis_thr:
                continue
            cx = int(pts[idx][0] + offset[0])
            cy = int(pts[idx][1] + offset[1])
            pygame.draw.circle(tmp, joint_color, (cx, cy), joint_r)

        # ── 5) 블렌딩 ────────────────────────────────────────────
        if alpha < 255:
            tmp.set_alpha(alpha)
            surface.blit(tmp, (px, py))

    # ------------------------------------------------------------------ #
    #  좌표 변환                                                          #
    # ------------------------------------------------------------------ #
    def _landmarks_to_pixels(
        self, landmarks: np.ndarray, panel_rect: Tuple[int, int, int, int],
    ) -> Optional[np.ndarray]:
        """(33,4) 랜드마크를 panel_rect 내부 픽셀 좌표로 변환합니다."""
        px, py, pw, ph = panel_rect

        pad_x = int(pw * 0.10)
        pad_y = int(ph * 0.06)
        draw_x = px + pad_x
        draw_y = py + pad_y
        draw_w = pw - pad_x * 2
        draw_h = ph - pad_y * 2

        vis_mask = landmarks[:, 3] > self.vis_thr
        if not np.any(vis_mask):
            return None

        xs = landmarks[vis_mask, 0]
        ys = landmarks[vis_mask, 1]
        x_min, x_max = float(xs.min()), float(xs.max())
        y_min, y_max = float(ys.min()), float(ys.max())

        x_range = max(x_max - x_min, 1e-6)
        y_range = max(y_max - y_min, 1e-6)

        scale = min(draw_w / x_range, draw_h / y_range)
        fig_w = x_range * scale
        fig_h = y_range * scale
        off_x = draw_x + (draw_w - fig_w) / 2
        off_y = draw_y + (draw_h - fig_h) / 2

        pts = np.zeros((33, 2), dtype=np.float64)
        for i in range(33):
            nx = (landmarks[i, 0] - x_min) / x_range
            ny = (landmarks[i, 1] - y_min) / y_range
            pts[i, 0] = off_x + nx * fig_w
            pts[i, 1] = off_y + ny * fig_h

        return pts

    # ------------------------------------------------------------------ #
    #  머리 그리기                                                        #
    # ------------------------------------------------------------------ #
    def _draw_head(
        self, surface, pts, landmarks, shoulder_width,
        palette, offset, *,
        radius_add: int = 0,
        color_override=None,
        alpha_mult: float = 1.0,
    ):
        """코·어깨 기반 위치에 머리(원)를 그립니다."""
        import pygame

        if landmarks[_NOSE][3] < self.vis_thr:
            return

        head_color = color_override or palette.get("head", (100, 220, 255))
        if alpha_mult < 1.0:
            head_color = (*head_color, int(255 * alpha_mult))

        # 머리 중심: 코 위치
        cx = int(pts[_NOSE][0] + offset[0])
        cy = int(pts[_NOSE][1] + offset[1])

        # 머리 반지름: 어깨 너비의 ~28%
        radius = max(int(shoulder_width * 0.28) + radius_add, 6)

        pygame.draw.circle(surface, head_color, (cx, cy), radius)

    # ------------------------------------------------------------------ #
    #  몸통 / 팔 / 다리 세그먼트 그리기                                     #
    # ------------------------------------------------------------------ #
    def _draw_body_segments(
        self, surface, pts, landmarks, base_thick,
        palette, offset, *,
        width_add: int = 0,
        color_override=None,
        alpha_mult: float = 1.0,
    ):
        """_BODY_SEGMENTS 에 정의된 부위별 세그먼트를 둥근 선으로 그립니다."""
        import pygame

        for src, dst, thick_ratio, part_key in _BODY_SEGMENTS:
            if landmarks[src][3] < self.vis_thr or landmarks[dst][3] < self.vis_thr:
                continue

            color = color_override or palette.get(part_key, (80, 180, 255))
            if alpha_mult < 1.0:
                color = (*color, int(255 * alpha_mult))

            thickness = max(int(base_thick * (thick_ratio / 0.22)) + width_add, 2)

            p1 = (int(pts[src][0] + offset[0]), int(pts[src][1] + offset[1]))
            p2 = (int(pts[dst][0] + offset[0]), int(pts[dst][1] + offset[1]))

            # 둥근 끝(round cap) 선분 = 두꺼운 line + 양 끝에 원
            pygame.draw.line(surface, color, p1, p2, thickness)
            cap_r = thickness // 2
            pygame.draw.circle(surface, color, p1, cap_r)
            pygame.draw.circle(surface, color, p2, cap_r)

        # 몸통 채우기 (어깨·골반 4 꼭짓점 폴리곤)
        torso_indices = [_L_SHOULDER, _R_SHOULDER, _R_HIP, _L_HIP]
        if all(landmarks[i][3] >= self.vis_thr for i in torso_indices):
            color = color_override or palette.get("torso", (80, 180, 255))
            if alpha_mult < 1.0:
                color = (*color, int(255 * alpha_mult))

            torso_pts = [
                (int(pts[i][0] + offset[0]), int(pts[i][1] + offset[1]))
                for i in torso_indices
            ]
            pygame.draw.polygon(surface, color, torso_pts)
