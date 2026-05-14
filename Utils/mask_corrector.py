import cv2
import numpy as np

class MaskCorrector:
    
    BILATERAL_PAIRS = {
        'brows': (2, 3),
        'eyes': (4, 5),
        'ears': (7, 8),
    }
    
    LIPS_UPPER = 12
    LIPS_LOWER = 13
    NOSE_ID = 10
    
    def __init__(self):
        self.corrections_log = []
    
    def correct_parsing_mask(self, parsing_mask, frame_shape):
        H, W = frame_shape[:2]
        corrected_mask = parsing_mask.copy()
        self.corrections_log = []
        
        # Calcola il centroide del naso una volta sola
        nose_center_x = self._get_nose_centroid_x(parsing_mask, W)
        
        # SCAMBIATI: right_id diventa left, left_id diventa right
        for pair_name, (left_id, right_id) in self.BILATERAL_PAIRS.items():
            corrected_mask = self._correct_bilateral_pair(
                corrected_mask, right_id, left_id, pair_name, nose_center_x
            )
        
        corrected_mask = self._correct_lips(corrected_mask, frame_shape)
        
        return corrected_mask, self.corrections_log
    
    def _get_nose_centroid_x(self, mask, frame_width):
        """Calcola la coordinata x del centroide del naso.
        Se il naso non è presente, usa il centro del frame."""
        nose_mask = (mask == self.NOSE_ID).astype(np.uint8)
        
        if np.sum(nose_mask) == 0:
            return frame_width // 2
        
        nose_blobs = self._get_connected_components(nose_mask)
        
        if len(nose_blobs) == 0:
            return frame_width // 2
        
        main_nose = max(nose_blobs, key=lambda b: b['area'])
        return main_nose['cx']
    
    def _correct_bilateral_pair(self, mask, left_id, right_id, pair_name, nose_center_x):
        left_mask = (mask == left_id).astype(np.uint8)
        right_mask = (mask == right_id).astype(np.uint8)
        
        left_blobs = self._get_connected_components(left_mask)
        right_blobs = self._get_connected_components(right_mask)
        
        if len(left_blobs) == 0 and len(right_blobs) > 0:
            mask = self._handle_missing_side(mask, right_blobs, right_id, left_id, 
                                             nose_center_x, pair_name, is_left_missing=True)
        
        elif len(right_blobs) == 0 and len(left_blobs) > 0:
            mask = self._handle_missing_side(mask, left_blobs, left_id, right_id,
                                             nose_center_x, pair_name, is_left_missing=False)
        
        elif len(left_blobs) > 0 and len(right_blobs) > 0:
            mask = self._handle_border_violations(mask, left_blobs, right_blobs,
                                                  left_id, right_id, nose_center_x, pair_name)
        
        return mask
    
    def _correct_lips(self, mask, frame_shape):
        upper_mask = (mask == self.LIPS_UPPER).astype(np.uint8)
        lower_mask = (mask == self.LIPS_LOWER).astype(np.uint8)
        
        upper_blobs = self._get_connected_components(upper_mask)
        lower_blobs = self._get_connected_components(lower_mask)
        
        if len(upper_blobs) > 0 and len(lower_blobs) > 0:
            mask = self._correct_lips_both_present(mask, upper_blobs, lower_blobs)
        
        elif len(upper_blobs) > 0 and len(lower_blobs) == 0:
            mask = self._correct_lips_single_present(mask, upper_blobs, self.LIPS_UPPER, 
                                                     self.LIPS_LOWER, is_upper=True)
        
        elif len(lower_blobs) > 0 and len(upper_blobs) == 0:
            mask = self._correct_lips_single_present(mask, lower_blobs, self.LIPS_LOWER,
                                                     self.LIPS_UPPER, is_upper=False)
        
        return mask
    
    def _correct_lips_both_present(self, mask, upper_blobs, lower_blobs):
        main_upper = max(upper_blobs, key=lambda b: b['area'])
        main_lower = max(lower_blobs, key=lambda b: b['area'])
        
        upper_cy = main_upper['cy']
        lower_cy = main_lower['cy']
        
        if upper_cy > lower_cy:
            temp_mask = (mask == self.LIPS_UPPER)
            mask[mask == self.LIPS_LOWER] = self.LIPS_UPPER
            mask[temp_mask] = self.LIPS_LOWER
            
            self.corrections_log.append({
                'type': 'lips_swapped',
                'pair': 'lips',
                'from_id': f'{self.LIPS_UPPER},{self.LIPS_LOWER}',
                'to_id': f'{self.LIPS_LOWER},{self.LIPS_UPPER}',
                'blob_area': main_upper['area'] + main_lower['area'],
                'reason': 'Upper lip was below lower lip - swapped all regions'
            })
        
        else:
            for blob in upper_blobs:
                if blob['cy'] > lower_cy:
                    mask[blob['mask']] = self.LIPS_LOWER
                    self.corrections_log.append({
                        'type': 'lips_position_fix',
                        'pair': 'lips',
                        'from_id': self.LIPS_UPPER,
                        'to_id': self.LIPS_LOWER,
                        'blob_area': blob['area'],
                        'reason': f'Upper lip blob below lower lip (cy={blob["cy"]:.1f} > {lower_cy:.1f})'
                    })
            
            for blob in lower_blobs:
                if blob['cy'] < upper_cy:
                    mask[blob['mask']] = self.LIPS_UPPER
                    self.corrections_log.append({
                        'type': 'lips_position_fix',
                        'pair': 'lips',
                        'from_id': self.LIPS_LOWER,
                        'to_id': self.LIPS_UPPER,
                        'blob_area': blob['area'],
                        'reason': f'Lower lip blob above upper lip (cy={blob["cy"]:.1f} < {upper_cy:.1f})'
                    })
        
        return mask
    
    def _correct_lips_single_present(self, mask, present_blobs, present_id, missing_id, is_upper):
        if len(present_blobs) > 1:
            main_blob = max(present_blobs, key=lambda b: b['area'])
            reference_y = main_blob['cy']
            
            for blob in present_blobs:
                if blob['id'] == main_blob['id']:
                    continue
                
                vertical_distance = blob['cy'] - reference_y
                
                if is_upper:
                    if vertical_distance > main_blob['h'] * 0.5:
                        mask[blob['mask']] = missing_id
                        self.corrections_log.append({
                            'type': 'lips_missing_reassignment',
                            'pair': 'lips',
                            'from_id': present_id,
                            'to_id': missing_id,
                            'blob_area': blob['area'],
                            'reason': f'Upper lip blob too far below main (Δy={vertical_distance:.1f}px)'
                        })
                else:
                    if vertical_distance < -main_blob['h'] * 0.5:
                        mask[blob['mask']] = missing_id
                        self.corrections_log.append({
                            'type': 'lips_missing_reassignment',
                            'pair': 'lips',
                            'from_id': present_id,
                            'to_id': missing_id,
                            'blob_area': blob['area'],
                            'reason': f'Lower lip blob too far above main (Δy={vertical_distance:.1f}px)'
                        })
        
        return mask
    
    def _get_connected_components(self, binary_mask):
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary_mask, connectivity=8
        )
        
        blobs = []
        for i in range(1, num_labels):
            blobs.append({
                'id': i,
                'area': stats[i, cv2.CC_STAT_AREA],
                'x': stats[i, cv2.CC_STAT_LEFT],
                'y': stats[i, cv2.CC_STAT_TOP],
                'w': stats[i, cv2.CC_STAT_WIDTH],
                'h': stats[i, cv2.CC_STAT_HEIGHT],
                'cx': centroids[i][0],
                'cy': centroids[i][1],
                'mask': (labels == i)
            })
        
        return blobs
    
    def _handle_missing_side(self, mask, present_blobs, present_id, missing_id,
                            nose_center_x, pair_name, is_left_missing):
        present_blobs.sort(key=lambda b: b['area'], reverse=True)
        
        for blob in present_blobs:
            blob_center_x = blob['cx']
            
            if is_left_missing:
                # Se manca il lato SINISTRO (left), cerco blob a SINISTRA del naso
                if blob_center_x < nose_center_x:
                    mask[blob['mask']] = missing_id
                    self.corrections_log.append({
                        'type': 'missing_side_reassignment',
                        'pair': pair_name,
                        'from_id': present_id,
                        'to_id': missing_id,
                        'blob_area': blob['area'],
                        'reason': f'Right {pair_name} found on left side (cx={blob_center_x:.1f} < nose_cx={nose_center_x:.1f})'
                    })
            else:
                # Se manca il lato DESTRO (right), cerco blob a DESTRA del naso
                if blob_center_x > nose_center_x:
                    mask[blob['mask']] = missing_id
                    self.corrections_log.append({
                        'type': 'missing_side_reassignment',
                        'pair': pair_name,
                        'from_id': present_id,
                        'to_id': missing_id,
                        'blob_area': blob['area'],
                        'reason': f'Left {pair_name} found on right side (cx={blob_center_x:.1f} > nose_cx={nose_center_x:.1f})'
                    })
        
        return mask
    
    def _handle_border_violations(self, mask, left_blobs, right_blobs,
                                  left_id, right_id, nose_center_x, pair_name):
        # Blob LEFT che attraversano a DESTRA → diventano RIGHT
        for blob in left_blobs:
            if blob['cx'] > nose_center_x:
                mask[blob['mask']] = right_id
                self.corrections_log.append({
                    'type': 'border_violation',
                    'pair': pair_name,
                    'from_id': left_id,
                    'to_id': right_id,
                    'blob_area': blob['area'],
                    'reason': f'Left {pair_name} crossed to right side (cx={blob["cx"]:.1f} > nose_cx={nose_center_x:.1f})'
                })
        
        # Blob RIGHT che attraversano a SINISTRA → diventano LEFT
        for blob in right_blobs:
            if blob['cx'] < nose_center_x:
                mask[blob['mask']] = left_id
                self.corrections_log.append({
                    'type': 'border_violation',
                    'pair': pair_name,
                    'from_id': right_id,
                    'to_id': left_id,
                    'blob_area': blob['area'],
                    'reason': f'Right {pair_name} crossed to left side (cx={blob["cx"]:.1f} < nose_cx={nose_center_x:.1f})'
                })
        
        return mask
    
    def get_corrections_summary(self):
        if not self.corrections_log:
            return "Nessuna correzione necessaria"
        
        summary = f"Applicate {len(self.corrections_log)} correzioni:\n"
        for i, corr in enumerate(self.corrections_log, 1):
            summary += f"  {i}. {corr['reason']} (area={corr['blob_area']}px)\n"
        
        return summary