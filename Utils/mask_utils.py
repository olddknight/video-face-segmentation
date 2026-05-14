import cv2
import numpy as np
from config import Config

COLOR_MAP = {
    1:  (230, 25, 75),      # skin - Viola/Porpora (BGR: blu-rosso dominanti)
    2:  (60, 180, 75),      # l_brow - Verde oliva
    3:  (0, 130, 200),      # r_brow - Arancione (BGR: rosso-verde)
    4:  (255, 225, 25),     # l_eye - Ciano chiaro (BGR: giallo-verde-blu)
    5:  (245, 130, 48),     # r_eye - Arancione chiaro
    6:  (145, 30, 180),     # eye_g - Viola intenso
    7:  (70, 240, 240),     # l_ear - Giallo brillante (BGR: ciano invertito)
    8:  (240, 50, 230),     # r_ear - Magenta/Rosa
    9:  (210, 245, 60),     # ear_r - Lime/Verde chiaro
    10: (250, 190, 190),    # nose - Rosa chiaro/Malva
    11: (0, 128, 128),      # mouth - Verde acqua scuro/Teal
    12: (230, 190, 255),    # u_lip - Lavanda chiaro
    13: (170, 110, 40),     # l_lip - Marrone/Terra
    14: (255, 250, 200),    # neck - Beige/Crema chiaro
    15: (128, 0, 0),        # neck_l - Bordeaux/Rosso scuro
    16: (170, 255, 195),    # hair - Verde menta chiaro
    17: (128, 128, 0),      # body - Verde acqua/Teal (BGR: giallo-verde)
    18: (0, 0, 0),          # background - Nero/Grigio molto scuro
}

_COLOR_LUT = np.zeros((256, 3), dtype=np.uint8)
for k, color in COLOR_MAP.items():
    _COLOR_LUT[int(k)] = np.array(color, dtype=np.uint8)

def apply_colormap_to_mask(mask):
    mask_u8 = mask.astype(np.uint8, copy=False)
    return _COLOR_LUT[mask_u8]

def blend_image_with_mask(image, mask_vis, alpha=0.5):
    return cv2.addWeighted(image, 1 - alpha, mask_vis, alpha, 0)