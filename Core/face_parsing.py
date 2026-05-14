import sys
import os
import torch
import cv2
import numpy as np
from torchvision import transforms, models
from PIL import Image
from config import Config
from scipy.ndimage import distance_transform_edt
from Utils.mask_corrector import MaskCorrector

def patch_weights_enum():
    original_get_state_dict = models.WeightsEnum.get_state_dict
    
    def patched_get_state_dict(self, *args, **kwargs):
        kwargs.pop('check_hash', None)
        return original_get_state_dict(self, *args, **kwargs)
    
    models.WeightsEnum.get_state_dict = patched_get_state_dict

patch_weights_enum()
sys.path.append(os.path.join(Config.BASE_DIR, "face-parsing-main"))
from models.bisenet import BiSeNet

class FaceParser:
    def __init__(self, bisenet_checkpoint=None):
        bisenet_checkpoint = bisenet_checkpoint or Config.BISENET_CHECKPOINT
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if not os.path.exists(bisenet_checkpoint):
            raise FileNotFoundError(f"Checkpoint non trovato: {bisenet_checkpoint}")
        
        self.bisenet = BiSeNet(num_classes=Config.BISENET_NUM_CLASSES, backbone_name=Config.BISENET_BACKBONE)
        self.bisenet.load_state_dict(torch.load(bisenet_checkpoint, map_location=self.device))
        self.bisenet.to(self.device).eval()
        
        self._half = False
        if self.device == "cuda" and Config.USE_HALF_PRECISION:
            torch.backends.cudnn.benchmark = Config.CUDNN_BENCHMARK
            self.bisenet = self.bisenet.half()
            self._half = True
        
        self.transform = transforms.Compose([
            transforms.Resize((Config.BISENET_INPUT_SIZE, Config.BISENET_INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])
        
        self.mask_corrector = MaskCorrector()
    
    def parse_face_region(self, frame, sam_mask, face_bbox, sam_face_mask):
        if face_bbox is None or sam_face_mask is None:
            return sam_mask
        
        if not self._is_valid_bbox(face_bbox):
            return sam_mask
        
        x1, y1, x2, y2 = face_bbox
        face_crop = frame[y1:y2, x1:x2]
        
        if face_crop.size == 0:
            return sam_mask
        
        parsing = self._parse_crop(face_crop)
        if parsing is None:
            return sam_mask
        
        combined_mask = self._integrate_bisenet_with_sam(sam_mask, parsing, face_bbox, sam_face_mask, frame.shape)
        final_mask = self._expand_facial_masks(combined_mask, sam_face_mask, frame.shape)
        
        corrected_mask, corrections = self.mask_corrector.correct_parsing_mask(
            final_mask, frame.shape
        )
        
        if corrections:
            print(f"[MASK CORRECTOR] {self.mask_corrector.get_corrections_summary()}")
        
        return corrected_mask
    
    def _is_valid_bbox(self, bbox):
        x1, y1, x2, y2 = bbox
        return (x2 > x1 and y2 > y1 and 
                (x2 - x1) >= Config.MIN_VALID_BBOX_SIZE and 
                (y2 - y1) >= Config.MIN_VALID_BBOX_SIZE)
    
    def _parse_crop(self, crop):
        try:
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            tensor = self.transform(Image.fromarray(crop_rgb)).unsqueeze(0)
            
            if self.device == "cuda":
                tensor = tensor.cuda()
                if self._half:
                    tensor = tensor.half()
            
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=(self.device == "cuda")):
                output = self.bisenet(tensor)[0]
                pred = output.squeeze(0).argmax(0)
            
            return pred.cpu().numpy()
            
        except Exception as e:
            print(f"[BiSeNet] Errore parsing: {e}")
            return None
        finally:
            if self.device == "cuda":
                torch.cuda.empty_cache()
    
    def _integrate_bisenet_with_sam(self, sam_mask, parsing, bbox, sam_face_mask, frame_shape):
        x1, y1, x2, y2 = bbox
        
        parsing_resized = cv2.resize(
            parsing.astype(np.uint8),
            (x2 - x1, y2 - y1),
            interpolation=cv2.INTER_NEAREST
        )
        
        sam_face_crop = sam_face_mask[y1:y2, x1:x2]
        result = sam_mask.copy()
        
        for class_id in Config.FACE_CLASSES:
            class_mask = (parsing_resized == class_id)
            valid_mask = class_mask & sam_face_crop
            
            if np.any(valid_mask):
                result[y1:y2, x1:x2][valid_mask] = class_id
        
        return result
    
    def _expand_facial_masks(self, mask, sam_face_mask, frame_shape):
        H, W = frame_shape[:2]
        
        empty_in_face = sam_face_mask & ~np.isin(mask, Config.FACE_CLASSES)
        
        if not np.any(empty_in_face):
            return mask
        
        facial_classes_present = [c for c in Config.FACE_CLASSES if np.any(mask == c)]
        
        if not facial_classes_present:
            return mask
        
        distances = np.full((len(facial_classes_present), H, W), np.inf)
        
        for i, class_id in enumerate(facial_classes_present):
            class_mask = (mask == class_id).astype(np.uint8)
            if np.any(class_mask):
                distances[i] = distance_transform_edt(1 - class_mask)
        
        nearest_class_idx = np.argmin(distances, axis=0)
        result = mask.copy()
        
        empty_coords = np.argwhere(empty_in_face)
        for y, x in empty_coords:
            result[y, x] = facial_classes_present[nearest_class_idx[y, x]]
        
        return result
    
    def clear_cache(self):
        if self.device == "cuda":
            torch.cuda.empty_cache()