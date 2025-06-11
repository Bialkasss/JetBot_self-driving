import cv2
import onnxruntime as rt

from pathlib import Path
import yaml
import numpy as np
import PIL.Image
from PUTDriver import PUTDriver, gstreamer_pipeline
import torchvision.transforms as transforms


class AI:
    def __init__(self, config: dict):
        self.path = config['model']['path']

        self.sess = rt.InferenceSession(self.path, providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])
 
        self.output_name = self.sess.get_outputs()[0].name
        self.input_name = self.sess.get_inputs()[0].name

    def preprocess(self, img):
        """
        Preprocessing function for robot inference.
        Applies the same preprocessing as training: resize, CLAHE, Gaussian blur, and ImageNet normalization.
        
        Args:
            img: Input image from robot camera (BGR format from OpenCV)
        
        Returns:
            Preprocessed image tensor ready for model inference (C, H, W format)
        """
        
        # Resize image (assuming same image_size as training, e.g., 64x64)
        image_size = 64  # Should match your training image_size ==in left its 128
        img = cv2.resize(img, (image_size, image_size))

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

    def postprocess(self, detections: np.ndarray) -> np.ndarray:
        ##TODO: prepare your outputs
        # The model outputs [left, forward] as a (1, 2) numpy array.
        # The main loop expects forward, left = ai.predict(image),
        # meaning the function should return [forward, left].
        # detections[0, 0] is 'left', detections[0, 1] is 'forward'.
        print(f"Detections {detections}")
        forward_output = detections[0, 1]
        left_output = detections[0, 0]
        forward_output = min(forward_output, 0.99)
        left_output = min(left_output, 0.99)
        left_output = max(left_output, -0.99)
        forward_output = max(forward_output, -0.99)

        # Return as a numpy array [forward, left]
        outputs = np.array([forward_output, left_output], dtype=np.float32)

        return outputs

    def predict(self, img: np.ndarray) -> np.ndarray:
        inputs = self.preprocess(img)

        assert inputs.dtype == np.float32
        assert inputs.shape == (1, 3, 224, 224)
        
        detections = self.sess.run([self.output_name], {self.input_name: inputs})[0]
        outputs = self.postprocess(detections)

        assert outputs.dtype == np.float32
        assert outputs.shape == (2,)
        assert outputs.max() < 1.0
        assert outputs.min() > -1.0

        return outputs


def main():
    with open("config.yml", "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    print("config loaded")

    driver = PUTDriver(config=config)
    print("driver loaded")
    ai = AI(config=config)
    print("ai loaded")

    video_capture = cv2.VideoCapture(gstreamer_pipeline(flip_method=0, display_width=224, display_height=224), cv2.CAP_GSTREAMER)

    # model warm-up
    ret, image = video_capture.read()
    if not ret:
        print(f'No camera')
        return
    
    _ = ai.predict(image)

    input('Robot is ready to ride. Press Enter to start...')

    forward, left = 0.0, 0.0
    while True:
        print(f'Forward: {forward:.4f}\tLeft: {left:.4f}')
        driver.update(forward, left)

        ret, image = video_capture.read()
        if not ret:
            print(f'No camera')
            break
        forward, left = ai.predict(image)


if __name__ == '__main__':
    main()

