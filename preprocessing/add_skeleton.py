import cv2
import mediapipe as mp
import numpy as np
import os

def overlay_skeleton(orig_video_path, silhouette_video_path, output_path):
    if not os.path.exists(orig_video_path):
        print(f"Error: Original video not found at {orig_video_path}")
        return
    if not os.path.exists(silhouette_video_path):
        print(f"Error: Silhouette video not found at {silhouette_video_path}")
        return

    cap_orig = cv2.VideoCapture(orig_video_path)
    cap_sil = cv2.VideoCapture(silhouette_video_path)

    # Output video properties (match silhouette video)
    width = int(cap_sil.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_sil.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap_sil.get(cv2.CAP_PROP_FPS)
    frame_count = int(min(cap_orig.get(cv2.CAP_PROP_FRAME_COUNT), cap_sil.get(cv2.CAP_PROP_FRAME_COUNT)))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Processing {frame_count} frames to add skeleton...")

    # MediaPipe Pose initialized
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    count = 0
    with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:
            
        while True:
            ret_orig, frame_orig = cap_orig.read()
            ret_sil, frame_sil = cap_sil.read()

            if not ret_orig or not ret_sil:
                break

            # Process original frame to get pose landmarks
            # Convert BGR to RGB before processing
            frame_orig_rgb = cv2.cvtColor(frame_orig, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_orig_rgb)

            # Draw the pose annotations on the silhouette frame
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame_sil,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )

            # Write the new frame to the output video
            out.write(frame_sil)

            count += 1
            if count % 100 == 0:
                print(f"Processed {count}/{frame_count} frames...")

    cap_orig.release()
    cap_sil.release()
    out.release()
    print(f"Successfully saved skeleton video to {output_path}")

if __name__ == "__main__":
    orig_video = "assets/videos/CJS1776217022.mp4"
    sil_video = "assets/videos/CJS1776217022_silhouette.mp4"
    output_video = "assets/videos/CJS1776217022_skeleton.mp4"
    
    os.makedirs(os.path.dirname(output_video), exist_ok=True)
    overlay_skeleton(orig_video, sil_video, output_video)
