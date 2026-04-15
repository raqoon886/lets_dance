import cv2
import numpy as np
import os
import argparse

def process_video(input_path, output_path):
    if not os.path.exists(input_path):
        print(f"Error: Input video not found at {input_path}")
        return

    cap = cv2.VideoCapture(input_path)

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Use mp4v codec for mp4 output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing {frame_count} frames...")

    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to HSV color space for better color segmentation
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define bounds for green screen. 
        # Hue for green is around 60 (OpenCV uses 0-179 for Hue)
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])

        # Create mask for the green background
        bg_mask = cv2.inRange(hsv, lower_green, upper_green)

        # Invert the mask to get the person
        person_mask = cv2.bitwise_not(bg_mask)

        # Refine the mask (remove noise)
        kernel = np.ones((3,3), np.uint8)
        person_mask = cv2.morphologyEx(person_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        person_mask = cv2.morphologyEx(person_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        # Create a white background frame
        result = np.full(frame.shape, 255, dtype=np.uint8)

        # Color the person with gray (128, 128, 128)
        result[person_mask > 0] = (128, 128, 128)

        out.write(result)
        
        count += 1
        if count % 100 == 0:
            print(f"Processed {count}/{frame_count} frames...")

    cap.release()
    out.release()
    print(f"Successfully saved silhouette video to {output_path}")

if __name__ == "__main__":
    input_video = "assets/videos/CJS1776217022.mp4"
    output_video = "assets/videos/CJS1776217022_silhouette.mp4"
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_video), exist_ok=True)
    
    process_video(input_video, output_video)
