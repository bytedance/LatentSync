import cv2
from typing import Optional
import os
import json
import torch
from tqdm import tqdm
from einops import rearrange
from ..utils.image_processor import ImageProcessor, load_fixed_mask


class AlignmentPipeline:
    _optional_components = []

    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def __call__(
            self,
            video_path: str,
            height: Optional[int] = None,
            width: Optional[int] = None):
        mask = "fix_mask"
        mask_image_path = "latentsync/utils/mask.png"

        mask_image = load_fixed_mask(height, mask_image_path)
        self.image_processor = ImageProcessor(height, mask=mask, device="cuda", mask_image=mask_image)

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print("Error: Could not open video.")
            return None

        file_name = os.path.splitext(os.path.basename(video_path))[0]
        folder_path = os.path.join(os.path.dirname(video_path), file_name)
        os.makedirs(folder_path, exist_ok=True)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_count = 0
        with tqdm(total=total_frames, desc="Processing frames", unit="frame") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face, box, affine_matrix = self.image_processor.affine_transform(frame_rgb)
                face = rearrange(face, "c h w -> h w c").to(torch.uint8).cpu().numpy()
                aligned_file_path = os.path.join(folder_path, f"{frame_count}.png")
                affine_path = os.path.join(folder_path, f"{frame_count}.json")

                cv2.imwrite(aligned_file_path, face[:, :, ::-1])
                with open(affine_path, "w") as f:
                    json.dump({"rot_mat": affine_matrix.tolist(), "box": box}, f)

                frame_count += 1
                pbar.update(1)

        cap.release()
        return True


if __name__ == "__main__":
    pipeline = AlignmentPipeline()
    pipeline(video_path="video_path", width=256, height=256)
