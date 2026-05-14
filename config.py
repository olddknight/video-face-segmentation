import os

class Config:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Model checkpoints
    SAM_CHECKPOINT = os.path.join(BASE_DIR, "segment-anything-main/sam_vit_h_4b8939.pth")
    BISENET_CHECKPOINT = os.path.join(BASE_DIR, "face-parsing-main/weights/resnet34.pt")
    
    # Output directories
    OUTPUT_DIR = "outputs"
    TEMP_DIR = os.path.join(OUTPUT_DIR, "temp")
    INPUT_FRAMES_DIR = os.path.join(TEMP_DIR, "input_frames")
    PROCESSED_FRAMES_DIR = os.path.join(TEMP_DIR, "processed_frames")
    CSV_DIR = os.path.join(OUTPUT_DIR, "analysis_csv")
    
    # ===== VIDEO SETTINGS =====
    JPEG_QUALITY = 95
    VIDEO_CODEC = 'mp4v'
    PROGRESS_UPDATE_INTERVAL = 5
    GC_INTERVAL = 50
    
    # ===== SEGMENTATION MODE =====
    USE_PREDICTOR_MODE = False  
    
    # ===== SEGMENTATION CLASSES =====
    BODY_CLASS = 17
    HAIR_CLASS = 16
    BACKGROUND_CLASS = 18
    FACE_CLASSES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    
    # ===== GPU OPTIMIZATION =====
    USE_HALF_PRECISION = True
    CUDNN_BENCHMARK = True
    MATMUL_PRECISION = 'medium'
    
    # ===== SAM PARAMETERS =====
    SAM_POINTS_PER_SIDE = 32
    SAM_PRED_IOU_THRESH = 0.88
    SAM_STABILITY_SCORE_THRESH = 0.92
    SAM_CROP_N_LAYERS = 1
    SAM_CROP_N_POINTS_DOWNSCALE = 2
    SAM_MIN_MASK_REGION_AREA = 100
    SAM_OVERLAP_THRESHOLD = 0.3
    SAM_DILATION_KERNEL_SIZE = 3
    
    # SAM Predictor Optimization
    PREDICTOR_INTERVAL = 10  
    PREDICTOR_MULTIMASK = True  
    PREDICTOR_STABILITY_THRESHOLD = 0.92  
    
    # ===== BISENET PARAMETERS =====
    BISENET_INPUT_SIZE = 512
    BISENET_NUM_CLASSES = 19
    BISENET_BACKBONE = 'resnet34'
    
    # ===== FACE DETECTION =====
    FACE_BBOX_SIZE = 512
    MIN_VALID_BBOX_SIZE = 100
    
    # ===== VISUALIZATION =====
    BLEND_ALPHA = 0.5
    
    @classmethod
    def ensure_directories(cls):
        """Crea tutte le directory necessarie"""
        for dir_path in [cls.OUTPUT_DIR, cls.TEMP_DIR, cls.INPUT_FRAMES_DIR, 
                         cls.PROCESSED_FRAMES_DIR, cls.CSV_DIR]:
            os.makedirs(dir_path, exist_ok=True)
