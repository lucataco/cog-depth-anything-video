# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
import os
import cv2
import torch
import tempfile
import subprocess
import numpy as np
from PIL import Image
import torch.nn.functional as F
from transformers import pipeline
from torchvision import transforms
from torchvision.transforms import Compose
from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""

    @torch.no_grad()
    def predict_depth(self, model, image):
        return model(image)["depth"]

    def predict(
        self,
        video: Path = Input(description="Input video"),
        encoder: str = Input(description="Model type", default="vits", choices=["vits", "vitb", "vitl"]),
    ) -> Path:
        """Run a single prediction on the model"""
        subprocess.run(["mkdir", "-p", "/tmp/frames/"])
        mapper = {"vits":"small","vitb":"base","vitl":"large"}
        to_tensor_transform = transforms.ToTensor()
        depth_anything = pipeline(task = "depth-estimation", model=f"nielsr/depth-anything-{mapper[encoder]}", device=0)

        transform = Compose([
            Resize(
                width=518,
                height=518,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])
        filename = str(video)
        print('Processing', filename)
        raw_video = cv2.VideoCapture(filename)
        frame_width, frame_height = int(raw_video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(raw_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_rate = int(raw_video.get(cv2.CAP_PROP_FPS))
        filename = os.path.basename(filename)
        count = 0
        while raw_video.isOpened():
            ret, raw_frame = raw_video.read()
            if not ret:
                break
            frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB) / 255.0
            frame_pil =  Image.fromarray((frame * 255).astype(np.uint8))
            frame = transform({'image': frame})['image']
            frame = torch.from_numpy(frame).unsqueeze(0).to('cuda')
            depth = to_tensor_transform(self.predict_depth(depth_anything, frame_pil))
            depth = F.interpolate(depth[None], (frame_height, frame_width), mode='bilinear', align_corners=False)[0, 0]
            depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
            depth = depth.cpu().numpy().astype(np.uint8)
            depth_color = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)
            cv2.imwrite(f'/tmp/frames/depth-{count}.png', depth_color)
            count += 1

        raw_video.release()
        # Convert all images in frames folder to a video
        subprocess.run(["ffmpeg", "-y", "-r", str(frame_rate), "-i", "/tmp/frames/depth-%d.png", "-vcodec", "libx264", "-crf", "25", "-pix_fmt", "yuv420p", "/tmp/output.mp4"])
        # cleanup
        subprocess.run(["rm", "-rf", "/tmp/frames"])
        return Path("/tmp/output.mp4")
