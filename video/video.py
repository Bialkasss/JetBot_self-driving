import torch
import cv2
import numpy as np
from pathlib import Path
import os
import pandas as pd
from typing import Dict, Tuple
import onnxruntime as ort


def load_model_and_data(model_path: str):
    """Load the trained model and test dataset"""
    print(f"Loading ONNX model from {model_path}")
    session = ort.InferenceSession(model_path)
    
    test_images = []
    test_labels = []
    

    src_files = Path("./data/eval").glob("*.csv")
    
    for src_file in src_files:
        dir_name, _ = os.path.splitext(src_file)
        print(f'Loading test data from {src_file}')
        data = pd.read_csv(src_file, header=None)
        
        for _, (idx, forward, sides) in data.iterrows():
            img_path = os.path.join(dir_name, str(int(idx)).zfill(4) + '.jpg')
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                test_images.append(img)
                test_labels.append((forward, sides))
    
    return session, test_images, test_labels


def preprocess_image(image: np.ndarray):
    
        image_size = 64  # Should match your training image_size ==in left its 128
        img = cv2.resize(image, (image_size, image_size))

        # Apply CLAHE enhancement
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(10, 10))

        y, u, v = cv2.split(img_yuv)
        y_clahe = clahe.apply(y)
        img_clahe = cv2.merge((y_clahe, u, v))
        img_clahe = cv2.cvtColor(img_clahe, cv2.COLOR_YUV2BGR)

        # Apply Gaussian blur
        img_blurred = cv2.GaussianBlur(img_clahe, (3, 3), 0)

        # Normalize to [0, 1] first
        img_blurred = img_blurred.astype(np.float32) / 255.0

        # Convert to PyTorch format (C, H, W)
        img_blurred = np.transpose(img_blurred, (2, 0, 1))
        

        return img_blurred


def create_signal_bars_opencv(left_right_signal: float, forward_signal: float, 
                             img_width: int, bar_height: int = 120) -> np.ndarray:
    """Create visualization bars using OpenCV only"""
    
    bars_img = np.zeros((bar_height, img_width, 3), dtype=np.uint8)
    
    bar_width_lr = img_width // 2 - 40
    bar_width_fwd = 40
    bar_thickness = 30
    
    lr_center_x = img_width // 4
    lr_center_y = bar_height // 3
    
    cv2.line(bars_img, (lr_center_x - bar_width_lr//2, lr_center_y), 
             (lr_center_x + bar_width_lr//2, lr_center_y), (100, 100, 100), 2)
    
    bar_length = int(abs(left_right_signal) * bar_width_lr / 2)
    if left_right_signal > 0:
        start_x = lr_center_x - bar_length
        end_x = lr_center_x
        color = (0, 0, 255)
    elif left_right_signal < 0:
        start_x = lr_center_x
        end_x = lr_center_x + bar_length
        color = (0, 255, 0)
    else:
        start_x = end_x = lr_center_x
        color = (128, 128, 128)
    
    if bar_length > 0:
        cv2.rectangle(bars_img, (start_x, lr_center_y - bar_thickness//2), 
                     (end_x, lr_center_y + bar_thickness//2), color, -1)
    
    cv2.putText(bars_img, "L", (lr_center_x - bar_width_lr//2 - 20, lr_center_y + 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(bars_img, "R", (lr_center_x + bar_width_lr//2 + 10, lr_center_y + 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(bars_img, f"L/R: {left_right_signal:.3f}", (10, 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    fwd_center_x = img_width * 3 // 4
    fwd_center_y = bar_height // 2
    bar_height_fwd = bar_height - 40
    
    cv2.line(bars_img, (fwd_center_x, fwd_center_y - bar_height_fwd//2), 
             (fwd_center_x, fwd_center_y + bar_height_fwd//2), (100, 100, 100), 2)
    
    bar_length = int(abs(forward_signal) * bar_height_fwd / 2)
    if forward_signal > 0:
        start_y = fwd_center_y - bar_length
        end_y = fwd_center_y
        color = (255, 0, 0)
    elif forward_signal < 0:
        start_y = fwd_center_y
        end_y = fwd_center_y + bar_length
        color = (0, 165, 255)
    else:
        start_y = end_y = fwd_center_y
        color = (128, 128, 128)
    
    if bar_length > 0:
        cv2.rectangle(bars_img, (fwd_center_x - bar_width_fwd//2, start_y), 
                     (fwd_center_x + bar_width_fwd//2, end_y), color, -1)
    
    cv2.putText(bars_img, "F", (fwd_center_x - 10, fwd_center_y - bar_height_fwd//2 - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(bars_img, "B", (fwd_center_x - 10, fwd_center_y + bar_height_fwd//2 + 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(bars_img, f"Fwd: {forward_signal:.3f}", (img_width//2 + 10, 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return bars_img


def create_prediction_video(model_path: str, output_path: str = "prediction_video.mp4", 
                          fps: int = 10, max_frames: int = None):
    """
    Create video showing model predictions with signal bars
    
    Args:
        model_path: Path to trained ONNX model
        output_path: Output video file path
        fps: Frames per second for output video
        max_frames: Maximum number of frames to process (None for all)
    """
    session, test_images, test_labels = load_model_and_data(model_path)
    
    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]
    print(f"Model input: {input_name}")
    print(f"Model outputs: {output_names}")
    
    if max_frames is not None:
        test_images = test_images[:max_frames]
        test_labels = test_labels[:max_frames]
    
    print(f"Processing {len(test_images)} frames")
    
    if len(test_images) == 0:
        print("No test images found!")
        return
    
    sample_img = test_images[0]
    img_height, img_width = sample_img.shape[:2]
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    bars_height = 120
    total_height = img_height + bars_height
    
    out = cv2.VideoWriter(output_path, fourcc, fps, (img_width, total_height))
    
    print("Generating video frames...")
    
    for i, (image, true_label) in enumerate(zip(test_images, test_labels)):
        if i % 50 == 0:
            print(f"Processing frame {i+1}/{len(test_images)}")
        
        
        img_numpy = np.expand_dims(preprocess_image(image), axis=0).astype(np.float32)        
        prediction = session.run(output_names, {input_name: img_numpy})
        
        pred_array = prediction[0][0]
        forward_pred = float(pred_array[0])
        left_right_pred = float(pred_array[1])
        
        bars_img = create_signal_bars_opencv(left_right_pred, forward_pred, img_width, bars_height)
        
        combined_frame = np.vstack([image, bars_img])
        
        true_fwd, true_lr = true_label
        text_lines = [
            f"Frame {i+1}",
            f"True L/R: {true_lr:.3f}, Pred: {left_right_pred:.3f}",
            f"True Fwd: {true_fwd:.3f}, Pred: {forward_pred:.3f}"
        ]
        
        y_offset = 30
        for line in text_lines:
            cv2.putText(combined_frame, line, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 25
        
        combined_frame_bgr = cv2.cvtColor(combined_frame, cv2.COLOR_RGB2BGR)
        
        out.write(combined_frame_bgr)
    
    out.release()
    print(f"Video saved to: {output_path}")


if __name__ == "__main__":
    MODEL_PATH = "name_of_the_model.onnx"
    OUTPUT_VIDEO = "robotics_predictions.mp4"
    FPS = 10
    MAX_FRAMES = None
    
    create_prediction_video(
        model_path=MODEL_PATH,
        output_path=OUTPUT_VIDEO,
        fps=FPS,
        max_frames=MAX_FRAMES
    )