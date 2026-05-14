import os
import cv2
from config import Config

class VideoProcessor:
    def __init__(self):
        os.makedirs(Config.INPUT_FRAMES_DIR, exist_ok=True)
        os.makedirs(Config.PROCESSED_FRAMES_DIR, exist_ok=True)
    
    def extract_frames(self, video_path, progress_callback=None):
        vidObj = cv2.VideoCapture(video_path)
        if not vidObj.isOpened():
            raise Exception(f"Impossibile aprire il video: {video_path}")
        
        total_frames = int(vidObj.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = vidObj.get(cv2.CAP_PROP_FPS)
        width = int(vidObj.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vidObj.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        frame_paths = []
        jpeg_params = [cv2.IMWRITE_JPEG_QUALITY, Config.JPEG_QUALITY]
        count = 0
        
        while True:
            success, image = vidObj.read()
            if not success:
                break
            
            frame_path = os.path.join(Config.INPUT_FRAMES_DIR, f"frame{count:04d}.jpg")
            cv2.imwrite(frame_path, image, jpeg_params)
            frame_paths.append(frame_path)
            count += 1
            
            if progress_callback and (count % 5 == 0 or count == total_frames):
                progress_callback(count, total_frames, "Estrazione frame")
        
        vidObj.release()
        return frame_paths, fps, total_frames, width, height
    
    def reconstruct_video(self, output_path, fps, width, height, progress_callback=None):
        frame_list = sorted(
            [f for f in os.listdir(Config.PROCESSED_FRAMES_DIR) if f.endswith(".jpg")], 
            key=lambda x: int(x[5:-4])
        )
        
        if not frame_list:
            raise Exception("Nessun frame processato trovato")
        
        if not output_path.endswith('.mp4'):
            output_path = output_path.rsplit('.', 1)[0] + '.mp4'
        
        fourcc = cv2.VideoWriter_fourcc(*Config.VIDEO_CODEC)
        video_out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not video_out.isOpened():
            raise Exception("Impossibile creare il video di output")
        
        for i, fname in enumerate(frame_list):
            frame = cv2.imread(os.path.join(Config.PROCESSED_FRAMES_DIR, fname))
            if frame is not None:
                video_out.write(frame)
            
            if progress_callback and (i + 1) % 10 == 0:
                progress_callback(i + 1, len(frame_list), "Scrittura video")
        
        video_out.release()