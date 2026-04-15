import cv2
import mediapipe as mp
import numpy as np
import os
import argparse

def process_video_onestop(input_path, output_path):
    if not os.path.exists(input_path):
        print(f"Error: Input video not found at {input_path}")
        return

    cap = cv2.VideoCapture(input_path)

    # Output video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Processing {frame_count} frames...")

    # Initialize MediaPipe Pose with segmentation enabled
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    count = 0
    with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            enable_segmentation=True) as pose:
            
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert BGR to RGB for MediaPipe processing
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            # Create a white background
            result_frame = np.full(frame.shape, 255, dtype=np.uint8)

            # If segmentation mask is available, apply the gray silhouette
            if results.segmentation_mask is not None:
                # The mask gives probability of the pixel being a person (0.0 to 1.0)
                # Create a binary mask condition: > 0.5 is person
                condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.5
                
                # Apply gray color where the person is
                gray_color = np.full(frame.shape, 128, dtype=np.uint8)
                result_frame = np.where(condition, gray_color, result_frame)

            # Draw the pose annotations on the result frame
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    result_frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )

            # Write the new frame to the output video
            out.write(result_frame)

            count += 1
            if count % 100 == 0:
                print(f"Processed {count}/{frame_count} frames...")

    cap.release()
    out.release()
    print(f"Successfully saved processed video to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract skeleton and create silhouette from video.")
    parser.add_argument("input_video", help="Path to the input video")
    parser.add_argument("--output_video", help="Path to the output video", default=None)
    
    args = parser.parse_args()
    
    input_video = args.input_video
    
    if args.output_video:
        output_video = args.output_video
    else:
        # Default output name based on input
        base, ext = os.path.splitext(input_video)
        output_video = f"{base}_processed.mp4"
        
    os.makedirs(os.path.dirname(output_video), exist_ok=True)
    process_video_onestop(input_video, output_video)
