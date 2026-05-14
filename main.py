import sys
import os
import cv2
import time
import gc
import numpy as np
import torch
import logging
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), 'Gui'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'Core'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'Utils'))

from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QApplication

from config import Config
from Gui.main_window import MainWindow
from Core.video_processor import VideoProcessor
from Core.segmentation import Segmentator
from Core.face_parsing import FaceParser
from Utils.mask_utils import apply_colormap_to_mask, blend_image_with_mask
from Utils.csv_writer import CSVWriter
from Utils.mask_saver import MaskSaver

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('segmentation.log'),
        logging.StreamHandler()
    ]
)


class ProcessingThread(QThread):
    progress_updated = pyqtSignal(int, int, str, str)
    finished_processing = pyqtSignal(bool, str)
    log_message = pyqtSignal(str) 
    
    def __init__(self, video_path, output_video_path, use_predictor=False):
        super().__init__()
        self.video_path = video_path
        self.output_video_path = output_video_path
        self.use_predictor_mode = use_predictor
        self.should_stop = False
        self.start_time = None
        self.logger = logging.getLogger(__name__)
        
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        csv_path = os.path.join(Config.CSV_DIR, f"{video_name}_analysis.csv")
        
        self.video_processor = VideoProcessor()
        self.segmentator = Segmentator()
        self.face_parser = FaceParser()
        self.csv_writer = CSVWriter(csv_path)
        self.mask_saver = MaskSaver(video_name)
    
    def stop_processing(self):
        self.should_stop = True
        self.logger.info("Stop richiesto")
    
    def run(self):
        try:
            self.log_message.emit("Inizio processamento...")
            self.logger.info(f"Elaborazione: {self.video_path}")
            self.start_time = time.time()
            
            frame_paths, fps, total_frames, width, height = self.video_processor.extract_frames(
                self.video_path, 
                lambda c, t, m: self._emit_progress(c, t, m)
            )
            
            if not frame_paths:
                raise Exception("Nessun frame estratto")
            
            self.logger.info(f"Estratti {len(frame_paths)} frame a {fps} FPS")
            
            processed_count = self._process_frames(frame_paths, total_frames)
            
            if self.should_stop:
                raise Exception("Elaborazione interrotta dall'utente")
            
            self.video_processor.reconstruct_video(
                self.output_video_path, fps, width, height,
                lambda c, t, m: self._emit_progress(c, t, m)
            )
            
            total_time = time.time() - self.start_time
            success_msg = f"Completato: {processed_count} frame in {self._format_time(total_time)}"
            self.logger.info(success_msg)
            self.finished_processing.emit(True, success_msg)
            
        except Exception as e:
            error_msg = f"Errore: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            self.finished_processing.emit(False, error_msg)
        finally:
            self._cleanup_resources()
    
    def _process_frames(self, frame_paths, total_frames):
        processed_count = 0
        jpeg_params = [cv2.IMWRITE_JPEG_QUALITY, Config.JPEG_QUALITY]
        
        for idx, frame_path in enumerate(frame_paths):
            if self.should_stop:
                break
            
            use_predictor = (
                self.use_predictor_mode and 
                (idx > 0) and 
                (idx % Config.PREDICTOR_INTERVAL != 0)
            )
            
            result = self._process_single_frame(frame_path, jpeg_params, idx, use_predictor)
            
            if result:
                output_path, blended = result
                cv2.imwrite(output_path, blended, jpeg_params)
                processed_count += 1
            
            if (idx + 1) % Config.PROGRESS_UPDATE_INTERVAL == 0 or (idx + 1) == total_frames:
                self._emit_progress(processed_count, total_frames, "Segmentazione")
            
            if (idx + 1) % Config.GC_INTERVAL == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        return processed_count
    
    def _process_single_frame(self, frame_path, jpeg_params, frame_idx, use_predictor):
        try:
            frame = cv2.imread(frame_path)
            if frame is None:
                self.logger.warning(f"Impossibile leggere: {frame_path}")
                return None
            
            frame_number = int(os.path.splitext(os.path.basename(frame_path))[0].replace('frame', ''))
            
            if self.use_predictor_mode:
                method = "SAM Predictor" if use_predictor else "SAM Generator"
            else:
                method = "SAM Generator (solo)"
            
            self.log_message.emit(f"[Frame {frame_number}] Processing with {method}...")
            
            sam_result = self.segmentator.segment_single_frame(frame, use_predictor=use_predictor)
            sam_mask, face_bbox, sam_face_mask = sam_result
            
            self.log_message.emit(f"[Frame {frame_number}] {method} OK - Face: {face_bbox is not None}")
            
            final_mask = self.face_parser.parse_face_region(frame, sam_mask, face_bbox, sam_face_mask)
            
            self.log_message.emit(f"[Frame {frame_number}] BiSeNet OK - Classes: {np.unique(final_mask)}")
            
            self.mask_saver.add_frame_mask(frame_number, final_mask)
            
            output_path = os.path.join(Config.PROCESSED_FRAMES_DIR, f"frame{frame_number:04d}.jpg")
            
            if final_mask is not None and np.any(final_mask):
                mask_vis = apply_colormap_to_mask(final_mask)
                blended = blend_image_with_mask(frame, mask_vis, alpha=Config.BLEND_ALPHA)
                self.csv_writer.write_frame_data(frame_number, final_mask)
                return (output_path, blended)
            
            return (output_path, frame)
            
        except Exception as e:
            self.logger.error(f"Errore frame {frame_path}: {e}")
            return None
    
    def _emit_progress(self, current, total, phase):
        elapsed = time.time() - self.start_time if self.start_time else 0
        remaining = ((total - current) * elapsed / current) if current > 0 and elapsed > 0 else 0
        
        message = f"{phase} - Frame: {current}/{total}"
        time_str = self._format_time(remaining) if remaining > 0 else "Calcolando..."
        
        self.progress_updated.emit(current, total, message, time_str)
    
    def _format_time(self, seconds):
        return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"
    
    def _cleanup_resources(self):
        self.logger.info("Cleanup risorse...")
        
        try:
            self.csv_writer.close()
            
            masks_file = self.mask_saver.save()
            if masks_file:
                self.log_message.emit(f"Maschere salvate in: {masks_file}")
            
            self.segmentator.clear_cache()
            self.face_parser.clear_cache()
        except Exception as e:
            self.logger.warning(f"Errore durante cleanup: {e}")
        
        self._delete_temp_frames()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        self.logger.info("Cleanup completato")
    
    def _delete_temp_frames(self):
        try:
            import shutil
            if os.path.exists(Config.INPUT_FRAMES_DIR):
                shutil.rmtree(Config.INPUT_FRAMES_DIR)
            if os.path.exists(Config.PROCESSED_FRAMES_DIR):
                shutil.rmtree(Config.PROCESSED_FRAMES_DIR)
            os.makedirs(Config.INPUT_FRAMES_DIR, exist_ok=True)
            os.makedirs(Config.PROCESSED_FRAMES_DIR, exist_ok=True)
        except Exception as e:
            self.logger.warning(f"Errore eliminazione frame temporanei: {e}")


class CLIProcessor:
    """Processore per modalità CLI senza GUI"""
    
    def __init__(self, video_path, use_predictor=False):
        self.video_path = video_path
        self.use_predictor_mode = use_predictor
        self.logger = logging.getLogger(__name__)
        
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        self.output_video_path = f"outputs/{video_name}_segmented.mp4"
        csv_path = os.path.join(Config.CSV_DIR, f"{video_name}_analysis.csv")
        
        self.video_processor = VideoProcessor()
        self.segmentator = Segmentator()
        self.face_parser = FaceParser()
        self.csv_writer = CSVWriter(csv_path)
        self.mask_saver = MaskSaver(video_name)
        self.start_time = None
    
    def process(self):
        try:
            print("="*60)
            print(f"Elaborazione video: {self.video_path}")
            print(f"Modalità: {'Generator + Predictor' if self.use_predictor_mode else 'Solo Generator'}")
            print("="*60)
            
            self.start_time = time.time()
            
            frame_paths, fps, total_frames, width, height = self.video_processor.extract_frames(
                self.video_path, 
                lambda c, t, m: self._print_progress(c, t, m)
            )
            
            if not frame_paths:
                raise Exception("Nessun frame estratto")
            
            print(f"\n✓ Estratti {len(frame_paths)} frame a {fps} FPS")
            
            processed_count = self._process_frames(frame_paths, total_frames)
            
            print(f"\n✓ Processati {processed_count} frame")
            
            self.video_processor.reconstruct_video(
                self.output_video_path, fps, width, height,
                lambda c, t, m: self._print_progress(c, t, m)
            )
            
            total_time = time.time() - self.start_time
            print(f"\n{'='*60}")
            print(f"✓ COMPLETATO in {self._format_time(total_time)}")
            print(f"Video salvato: {self.output_video_path}")
            print("="*60)
            
            return True
            
        except Exception as e:
            print(f"\n✗ ERRORE: {str(e)}")
            self.logger.error("Errore elaborazione", exc_info=True)
            return False
        finally:
            self._cleanup_resources()
    
    def _process_frames(self, frame_paths, total_frames):
        processed_count = 0
        jpeg_params = [cv2.IMWRITE_JPEG_QUALITY, Config.JPEG_QUALITY]
        
        for idx, frame_path in enumerate(frame_paths):
            use_predictor = (
                self.use_predictor_mode and 
                (idx > 0) and 
                (idx % Config.PREDICTOR_INTERVAL != 0)
            )
            
            result = self._process_single_frame(frame_path, jpeg_params, idx, use_predictor)
            
            if result:
                output_path, blended = result
                cv2.imwrite(output_path, blended, jpeg_params)
                processed_count += 1
            
            if (idx + 1) % Config.PROGRESS_UPDATE_INTERVAL == 0 or (idx + 1) == total_frames:
                self._print_progress(processed_count, total_frames, "Segmentazione")
            
            if (idx + 1) % Config.GC_INTERVAL == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        return processed_count
    
    def _process_single_frame(self, frame_path, jpeg_params, frame_idx, use_predictor):
        try:
            frame = cv2.imread(frame_path)
            if frame is None:
                return None
            
            frame_number = int(os.path.splitext(os.path.basename(frame_path))[0].replace('frame', ''))
            
            sam_result = self.segmentator.segment_single_frame(frame, use_predictor=use_predictor)
            sam_mask, face_bbox, sam_face_mask = sam_result
            
            final_mask = self.face_parser.parse_face_region(frame, sam_mask, face_bbox, sam_face_mask)
            
            self.mask_saver.add_frame_mask(frame_number, final_mask)
            
            output_path = os.path.join(Config.PROCESSED_FRAMES_DIR, f"frame{frame_number:04d}.jpg")
            
            if final_mask is not None and np.any(final_mask):
                mask_vis = apply_colormap_to_mask(final_mask)
                blended = blend_image_with_mask(frame, mask_vis, alpha=Config.BLEND_ALPHA)
                self.csv_writer.write_frame_data(frame_number, final_mask)
                return (output_path, blended)
            
            return (output_path, frame)
            
        except Exception as e:
            self.logger.error(f"Errore frame {frame_path}: {e}")
            return None
    
    def _print_progress(self, current, total, phase):
        percent = int((current / total) * 100) if total > 0 else 0
        bar_length = 40
        filled = int(bar_length * current / total) if total > 0 else 0
        bar = '█' * filled + '-' * (bar_length - filled)
        
        elapsed = time.time() - self.start_time if self.start_time else 0
        remaining = ((total - current) * elapsed / current) if current > 0 and elapsed > 0 else 0
        
        print(f"\r{phase}: |{bar}| {percent}% ({current}/{total}) - Tempo residuo: {self._format_time(remaining)}", end='', flush=True)
    
    def _format_time(self, seconds):
        return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"
    
    def _cleanup_resources(self):
        print("\n\nPulizia risorse...")
        
        try:
            self.csv_writer.close()
            masks_file = self.mask_saver.save()
            if masks_file:
                print(f"✓ Maschere salvate: {masks_file}")
            
            self.segmentator.clear_cache()
            self.face_parser.clear_cache()
        except Exception as e:
            self.logger.warning(f"Errore durante cleanup: {e}")
        
        self._delete_temp_frames()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    def _delete_temp_frames(self):
        try:
            import shutil
            if os.path.exists(Config.INPUT_FRAMES_DIR):
                shutil.rmtree(Config.INPUT_FRAMES_DIR)
            if os.path.exists(Config.PROCESSED_FRAMES_DIR):
                shutil.rmtree(Config.PROCESSED_FRAMES_DIR)
            os.makedirs(Config.INPUT_FRAMES_DIR, exist_ok=True)
            os.makedirs(Config.PROCESSED_FRAMES_DIR, exist_ok=True)
        except Exception as e:
            self.logger.warning(f"Errore eliminazione frame temporanei: {e}")


def main():
    Config.ensure_directories()
    
    parser = argparse.ArgumentParser(
        description='Video Face Analysis - Segmentazione video con SAM e BiSeNet',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi di utilizzo:
  # Modalità GUI
  python Code/main.py
  
  # Modalità CLI con solo Generator
  python Code/main.py video.mp4
  
  # Modalità CLI con Generator + Predictor
  python Code/main.py video.mp4 true
        """
    )
    
    parser.add_argument('video_path', nargs='?', help='Path del video da processare (opzionale, avvia GUI se omesso)')
    parser.add_argument('use_predictor', nargs='?', choices=['true', 'false'], default='false',
                       help='Abilita modalità Generator + Predictor (default: false)')
    
    args = parser.parse_args()
    
    # Modalità CLI
    if args.video_path:
        if not os.path.exists(args.video_path):
            print(f"✗ Errore: File non trovato: {args.video_path}")
            return 1
        
        use_predictor = args.use_predictor.lower() == 'true'
        
        processor = CLIProcessor(args.video_path, use_predictor=use_predictor)
        success = processor.process()
        
        return 0 if success else 1
    
    # Modalità GUI
    else:
        app = QApplication(sys.argv)
        window = MainWindow(ProcessingThread)
        window.show()
        return app.exec_()


if __name__ == '__main__':
    sys.exit(main())