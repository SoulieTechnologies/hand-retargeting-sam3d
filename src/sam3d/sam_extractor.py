import cv2
import numpy as np
import torch
import os
import sys

sys.path.insert(0, "/home/theo/code/sam-3d-body")

from notebook.utils import setup_sam_3d_body

class SAM3DExtractor:
    def __init__(self):
        print("Loading SAM 3D Body Model (DINOv3-H)...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.estimator = setup_sam_3d_body(hf_repo_id="facebook/sam-3d-body-dinov3")
        print(f"Model loaded successfully on {self.device}!")

    def process_video(self, video_path, output_npy_path="hand_kinematics.npy"):   
        print(f"Opening video: {video_path}")
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Error: Could not open video file at {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Total frames to process: {total_frames}")
        sequence_joints = []
        for frame_idx in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                print(f"Reached end of video at frame {frame_idx}")
                break
            
            # rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # # Dict with vertices, faces, keypoints
            # outputs = self.estimator.process_one_image(rgb_frame)
            # --- NEW VRAM SAVING CODE ---
            # Shrink the frame so the longest side is 640 pixels
            h, w = frame.shape[:2]
            scale = 640.0 / max(h, w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
            # ----------------------------

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            with torch.no_grad():
                outputs = self.estimator.process_one_image(rgb_frame)

            # --- CLEAR CACHE AFTER EVERY FRAME ---
            torch.cuda.empty_cache() 
            # -------------------------------------

            joints_3d = outputs[0]['pred_keypoints_3d'][21:63].reshape(2, 21, 3)

            sequence_joints.append(joints_3d)
            
            if frame_idx % 10 == 0:
                print(f"Processed frame {frame_idx}/{total_frames}")

        cap.release()
        kinematics_matrix = np.stack(sequence_joints)
        np.save(output_npy_path, kinematics_matrix)
        print(f"Success! Kinematics saved to {output_npy_path}")
        print(f"Final Data Shape: {kinematics_matrix.shape}")

if __name__ == "__main__":
    my_video = "/home/theo/code/hand-retargeting-sam3d/data/raw_videos/test_hand_video1.mp4" 
    
    extractor = SAM3DExtractor()
    extractor.process_video(my_video, "../data/sam3d_outputs/dexsuite_joints1.npy")
