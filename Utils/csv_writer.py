import csv
import os
import numpy as np
import threading
import cv2
from config import Config

REGION_NAMES = {
    1: "skin", 2: "l_brow", 3: "r_brow", 4: "l_eye", 5: "r_eye",
    6: "eye_g", 7: "l_ear", 8: "r_ear", 9: "ear_r", 10: "nose",
    11: "mouth", 12: "u_lip", 13: "l_lip", 14: "neck", 15: "neck_l",
    16: "hair", 17: "body", 18: "background"
}

class CSVWriter:
    def __init__(self, output_path="outputs/analysis_csv/analysis.csv"):
        self.output_path = output_path
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self._lock = threading.Lock()
        
        file_exists = os.path.exists(output_path)
        mode = 'a' if file_exists else 'w'
        self._csvfile = open(output_path, mode, newline='', encoding='utf-8')
        self._writer = csv.writer(self._csvfile)
        
        if not file_exists:
            self._writer.writerow([
                'frame_number',
                'region_id', 
                'region_name',
                'blob_id',
                'total_blobs',
                'blob_area',
                'total_region_area',
                'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2',
                'centroid_x', 'centroid_y'
            ])
    
    def write_frame_data(self, frame_number, parsing_mask):
        unique_regions = np.unique(parsing_mask)
        
        if unique_regions.size == 0:
            return 0
        
        rows_written = 0
        
        for region_id in unique_regions:
            region_mask = (parsing_mask == region_id).astype(np.uint8)
            num_labels, labels = cv2.connectedComponents(region_mask)
            
            total_region_area = np.sum(region_mask)
            
            for blob_id in range(1, num_labels):
                blob_mask = (labels == blob_id).astype(np.uint8)
                blob_area = np.sum(blob_mask)
                
                if blob_area == 0:
                    continue
                
                y_coords, x_coords = np.where(blob_mask > 0)
                bbox_x1, bbox_y1 = int(x_coords.min()), int(y_coords.min())
                bbox_x2, bbox_y2 = int(x_coords.max()), int(y_coords.max())
                
                M = cv2.moments(blob_mask)
                if M['m00'] != 0:
                    centroid_x = M['m10'] / M['m00']
                    centroid_y = M['m01'] / M['m00']
                else:
                    centroid_x = (bbox_x1 + bbox_x2) / 2
                    centroid_y = (bbox_y1 + bbox_y2) / 2
                
                row = [
                    frame_number,
                    int(region_id),
                    REGION_NAMES.get(int(region_id), f"unknown_{region_id}"),
                    blob_id,
                    num_labels - 1,
                    int(blob_area),
                    int(total_region_area),
                    bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                    f"{centroid_x:.2f}", f"{centroid_y:.2f}"
                ]
                
                self._writer.writerow(row)
                rows_written += 1
        
        with self._lock:
            self._csvfile.flush()
        
        return rows_written
    
    def close(self):
        if hasattr(self, '_csvfile') and self._csvfile:
            self._csvfile.close()