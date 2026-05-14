import os
import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from config import Config
from scipy.ndimage import distance_transform_edt, binary_fill_holes

class Segmentator:
    def __init__(self, sam_checkpoint=None):
        sam_checkpoint = sam_checkpoint or Config.SAM_CHECKPOINT
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if not os.path.exists(sam_checkpoint):
            raise FileNotFoundError(f"Checkpoint SAM non trovato: {sam_checkpoint}")
        
        if self.device == "cuda":
            torch.backends.cudnn.benchmark = Config.CUDNN_BENCHMARK
            torch.set_float32_matmul_precision(Config.MATMUL_PRECISION)
        
        self.sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint).to(self.device)
        self.sam.eval()
        
        if self.device == "cuda" and Config.USE_HALF_PRECISION:
            self.sam = self.sam.half()
        
        self.mask_generator = SamAutomaticMaskGenerator(
            model=self.sam,
            points_per_side=Config.SAM_POINTS_PER_SIDE,
            pred_iou_thresh=Config.SAM_PRED_IOU_THRESH,
            stability_score_thresh=Config.SAM_STABILITY_SCORE_THRESH,
            crop_n_layers=Config.SAM_CROP_N_LAYERS,
            crop_n_points_downscale_factor=Config.SAM_CROP_N_POINTS_DOWNSCALE,
            min_mask_region_area=Config.SAM_MIN_MASK_REGION_AREA,
        )
        
        self.predictor = SamPredictor(self.sam)
        self.cached_boxes = None
    
    def segment_single_frame(self, frame, use_predictor=False):
        with torch.cuda.amp.autocast(enabled=(self.device == "cuda")), torch.no_grad():
            if use_predictor and self.cached_boxes is not None:
                return self._process_frame_with_predictor(frame)
            else:
                return self._process_frame_with_generator(frame)

    def _process_frame_with_generator(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        H, W = frame.shape[:2]
        
        masks = self.mask_generator.generate(frame_rgb)
        
        if not masks:
            print("[SAM Generator] Nessuna maschera generata")
            return self._empty_result(H, W)
        
        sorted_masks = sorted(masks, key=lambda x: x['area'], reverse=True)
        print(f"[SAM Generator] Trovate {len(sorted_masks)} maschere")
        
        self._cache_boxes_from_masks(sorted_masks)
        
        result = self._process_masks_common(sorted_masks, H, W)
        
        if result is None:
            return self._empty_result(H, W)
        
        sam_result, face_bbox, face_mask, final_masks = result
        
        return sam_result, face_bbox, face_mask
    
    def _process_frame_with_predictor(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        H, W = frame.shape[:2]
        
        self.predictor.set_image(frame_rgb)
        
        masks = []
        for box_info in self.cached_boxes:
            box = box_info['box']
            
            pred_masks, scores, logits = self.predictor.predict(
                point_coords=None,
                point_labels=None,
                box=box,
                multimask_output=Config.PREDICTOR_MULTIMASK
            )
            
            if Config.PREDICTOR_MULTIMASK:
                best_idx = np.argmax(scores)
                mask_data = pred_masks[best_idx]
                score = scores[best_idx]
            else:
                mask_data = pred_masks[0]
                score = scores[0]
            
            if score < Config.PREDICTOR_STABILITY_THRESHOLD:
                continue
            
            mask_dict = {
                'segmentation': mask_data,
                'area': mask_data.sum(),
                'bbox': box.tolist(),
                'predicted_iou': float(score),
                'stability_score': float(score)
            }
            masks.append(mask_dict)
        
        if not masks:
            print("[SAM Predictor] Nessuna maschera valida generata, fallback a Generator")
            return self._process_frame_with_generator(frame)
        
        sorted_masks = sorted(masks, key=lambda x: x['area'], reverse=True)
        print(f"[SAM Predictor] Trovate {len(sorted_masks)} maschere")
        
        self._cache_boxes_from_masks(sorted_masks)
        
        result = self._process_masks_common(sorted_masks, H, W)
        
        if result is None:
            print("[SAM Predictor] Fallback a Generator per maschere insufficienti")
            return self._process_frame_with_generator(frame)
        
        sam_result, face_bbox, face_mask, final_masks = result
        
        return sam_result, face_bbox, face_mask
    
    def _process_masks_common(self, sorted_masks, H, W):
        filtered_masks = self._filter_overlapping_masks(sorted_masks)
        print(f"[SAM] Dopo filtraggio: {len(filtered_masks)}")
        
        if len(filtered_masks) < 4:
            print(f"[SAM] Maschere insufficienti: {len(filtered_masks)}")
            return None
        
        expanded_masks = self._expand_masks_to_fill_gaps(filtered_masks, H, W)
        
        primary_masks = expanded_masks[:4]
        secondary_masks = expanded_masks[4:]
        
        if secondary_masks:
            secondary_masks = self._consolidate_masks(secondary_masks)
            print(f"[SAM] Maschere secondarie consolidate: {len(secondary_masks)}")
        
        primary_masks, secondary_masks = self._validate_face_hair_position(
            primary_masks, secondary_masks
        )
        
        final_masks = self._absorb_secondary_masks(primary_masks, secondary_masks, H, W)
        
        background_mask = final_masks[0]['segmentation']
        body_mask = final_masks[1]['segmentation']
        
        face_mask, hair_mask, face_bbox = self._extract_face_and_hair(
            final_masks[2], final_masks[3], (H, W)
        )
        
        sam_result = self._build_sam_mask(H, W, body_mask, hair_mask)
        
        print(f"[SAM] Classi finali: {np.unique(sam_result)}")
        return sam_result, face_bbox, face_mask, final_masks
    
    def _validate_face_hair_position(self, primary_masks, secondary_masks):
        """
        Valida che le maschere 2 e 3 (faccia/capelli) siano sopra il corpo.
        Se il centroide è più basso del limite superiore del corpo E
        il punto più alto della maschera è più basso del limite del corpo,
        sostituisce con una maschera secondaria (SWAP BIDIREZIONALE).
        """
        if len(secondary_masks) < 2:
            print("[SAM] Maschere secondarie insufficienti per la validazione")
            return primary_masks, secondary_masks
        
        body_mask = primary_masks[1]['segmentation']
        body_ys, _ = np.where(body_mask)
        body_top = body_ys.min() if len(body_ys) > 0 else 0
        
        print(f"[SAM] Limite superiore corpo: y={body_top}")
        
        swap_mask_2 = False
        swap_mask_3 = False
        
        # Valida maschera 2
        mask_2_seg = primary_masks[2]['segmentation']
        mask_2_ys, mask_2_xs = np.where(mask_2_seg)
        if len(mask_2_ys) > 0:
            centroid_2_y = int(mask_2_ys.mean())
            top_2_y = mask_2_ys.min()
            print(f"[SAM] Maschera 2 - Centroide: y={centroid_2_y}, Punto più alto: y={top_2_y}")
            
            if centroid_2_y >= body_top and top_2_y >= body_top:
                print("[SAM] Maschera 2 è troppo bassa (centroide E punto più alto sotto corpo), richiede swap")
                swap_mask_2 = True
        
        # Valida maschera 3
        mask_3_seg = primary_masks[3]['segmentation']
        mask_3_ys, mask_3_xs = np.where(mask_3_seg)
        if len(mask_3_ys) > 0:
            centroid_3_y = int(mask_3_ys.mean())
            top_3_y = mask_3_ys.min()
            print(f"[SAM] Maschera 3 - Centroide: y={centroid_3_y}, Punto più alto: y={top_3_y}")
            
            if centroid_3_y >= body_top and top_3_y >= body_top:
                print("[SAM] Maschera 3 è troppo bassa (centroide E punto più alto sotto corpo), richiede swap")
                swap_mask_3 = True
        
        if swap_mask_2 and swap_mask_3:
            print("[SAM] Swap bidirezionale: maschera 2 ↔ secondary[0] e maschera 3 ↔ secondary[1]")
            
            old_mask_2 = primary_masks[2]
            old_mask_3 = primary_masks[3]
            
            primary_masks[2] = secondary_masks[len(secondary_masks) - 1]
            primary_masks[3] = secondary_masks[len(secondary_masks) - 2]
            
            secondary_masks[len(secondary_masks) - 1] = old_mask_2
            secondary_masks[len(secondary_masks) - 2] = old_mask_3
            
        elif swap_mask_2:
            print("[SAM] Swap bidirezionale: maschera 2 ↔ secondary[0]")
            
            old_mask_2 = primary_masks[2]
            primary_masks[2] = secondary_masks[len(secondary_masks) - 1]
            secondary_masks[len(secondary_masks) - 1] = old_mask_2
            
        elif swap_mask_3:
            print("[SAM] Swap bidirezionale: maschera 3 ↔ secondary[0]")
            
            old_mask_3 = primary_masks[3]
            primary_masks[3] = secondary_masks[len(secondary_masks) - 1]
            secondary_masks[len(secondary_masks) - 1] = old_mask_3
        
        else:
            print("[SAM] Maschere 2 e 3 sono in posizioni valide")
        
        return primary_masks, secondary_masks

    

    def _consolidate_masks(self, masks):
        """
        Consolida le maschere riempiendo buchi e smoothando bordi.
        Ogni maschera viene processata indipendentemente.
        """
        consolidated = []
        
        for mask in masks:
            seg = mask['segmentation']
            

            filled = binary_fill_holes(seg).astype(np.uint8)
            

            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            closed = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, kernel_close)
            

            kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            smoothed = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)
            

            new_area = int(smoothed.sum())
            

            if new_area < Config.SAM_MIN_MASK_REGION_AREA:
                continue
            
            consolidated_mask = mask.copy()
            consolidated_mask['segmentation'] = smoothed.astype(bool)
            consolidated_mask['area'] = new_area
            
            consolidated.append(consolidated_mask)
        
        return consolidated

    
    def _cache_boxes_from_masks(self, masks):
        self.cached_boxes = []
        
        for mask in masks:
            seg = mask['segmentation']
            if not np.any(seg):
                continue
            
            ys, xs = np.where(seg)
            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()
            
            box = np.array([x_min, y_min, x_max, y_max], dtype=np.float32)
            
            self.cached_boxes.append({
                'box': box,
                'area': mask['area']
            })
    
    def _empty_result(self, H, W):
        empty_mask = np.full((H, W), Config.BACKGROUND_CLASS, dtype=np.uint8)
        return empty_mask, None, None
    
    def _filter_overlapping_masks(self, sorted_masks):
        filtered = []
        tmp = []
        
        for i, mask in enumerate(sorted_masks):
            keep_mask = True
            current_seg = mask['segmentation']
            
            for j in range(i + 1, len(sorted_masks)):
                other_seg = sorted_masks[j]['segmentation']
                intersection = np.logical_and(current_seg, other_seg).sum()
                union = np.logical_or(current_seg, other_seg).sum()
                
                if union > 0 and (intersection / union) > Config.SAM_OVERLAP_THRESHOLD:
                    keep_mask = False
                    break
            
            if keep_mask:
                filtered.append(mask)

            if len(filtered) == 4:
                tmp = sorted_masks[i + 1:]
                tmp.sort(key=lambda x: x['area'], reverse=False)
                break

        for i, mask in enumerate(tmp):
            keep_mask = True
            current_seg = mask['segmentation']
            
            for j in range(i + 1, len(tmp)):
                other_seg = tmp[j]['segmentation']
                intersection = np.logical_and(current_seg, other_seg).sum()
                union = np.logical_or(current_seg, other_seg).sum()
                
                if union > 0 and (intersection / union) > Config.SAM_OVERLAP_THRESHOLD:
                    keep_mask = False
                    break
            
            if keep_mask:
                filtered.append(mask)

        return filtered
    
    def _expand_masks_to_fill_gaps(self, masks, H, W):
        mask_map = np.zeros((H, W), dtype=np.int32)
        
        for i, mask in enumerate(masks):
            mask_map[mask['segmentation']] = i + 1
        
        empty_pixels = (mask_map == 0)
        
        if not np.any(empty_pixels):
            return masks
        
        print(f"[SAM] Trovati {empty_pixels.sum()} pixel vuoti da riempire")
        
        distances = np.full((len(masks), H, W), np.inf)
        
        for i, mask in enumerate(masks):
            mask_binary = mask['segmentation'].astype(np.uint8)
            distances[i] = distance_transform_edt(1 - mask_binary)
        
        nearest_mask_idx = np.argmin(distances, axis=0)
        
        expanded_masks = []
        for i, mask in enumerate(masks):
            expanded_mask = mask.copy()
            new_segmentation = mask['segmentation'].copy()
            new_segmentation |= (empty_pixels & (nearest_mask_idx == i))
            
            expanded_mask['segmentation'] = new_segmentation
            expanded_mask['area'] = new_segmentation.sum()
            expanded_masks.append(expanded_mask)
        
        return expanded_masks
    
    def _absorb_secondary_masks(self, primary_masks, secondary_masks, H, W):
        if not secondary_masks:
            return primary_masks
        
        background_mask = primary_masks[0]['segmentation']
        target_masks = primary_masks[1:4]
        
        kernel = np.ones((Config.SAM_DILATION_KERNEL_SIZE, 
                          Config.SAM_DILATION_KERNEL_SIZE), 
                         np.uint8)
        
        # Calcola le bounding box delle maschere primarie (corpo, viso, capelli)
        primary_bboxes = []
        for target_mask in target_masks:
            seg = target_mask['segmentation']
            if np.any(seg):
                ys, xs = np.where(seg)
                bbox = (xs.min(), ys.min(), xs.max(), ys.max())
                primary_bboxes.append(bbox)
            else:
                primary_bboxes.append(None)
        
        for sec_mask in secondary_masks:
            sec_seg = sec_mask['segmentation']
            
            # Calcola il perimetro della maschera secondaria
            sec_uint8 = sec_seg.astype(np.uint8)
            eroded = cv2.erode(sec_uint8, kernel, iterations=1)
            perimeter = sec_uint8 - eroded
            perimeter_pixels = np.sum(perimeter > 0)
            
            # Conta pixel del perimetro confinanti con background
            background_contact = np.logical_and(perimeter, background_mask).sum()
            
            # Calcola percentuale di contatto con background
            if perimeter_pixels > 0:
                background_contact_ratio = background_contact / perimeter_pixels
            else:
                background_contact_ratio = 0.0
            
            # Condizione: se confina con background >= 30% assorbila nello sfondo
            if background_contact_ratio >= 0.30:
                print(f"[SAM] Maschera secondaria assorbita dallo sfondo "
                      f"(contatto background: {background_contact_ratio:.2%})")
                
                primary_masks[0]['segmentation'] |= sec_seg
                primary_masks[0]['area'] += sec_mask['area']
                continue
            
            # Altrimenti, procedi con l'assorbimento normale nelle maschere target
            candidates = []
            
            for i, target_mask in enumerate(target_masks):
                target_seg = target_mask['segmentation']
                dilated_target = cv2.dilate(target_seg.astype(np.uint8), 
                                            kernel, 
                                            iterations=1)
                
                overlap = np.logical_and(sec_seg, dilated_target).sum()
                
                if overlap > 0:
                    candidates.append((i, target_mask['area']))
            
            if candidates:
                best_target_idx = max(candidates, key=lambda x: x[1])[0]
                
                target_masks[best_target_idx]['segmentation'] |= sec_seg
                target_masks[best_target_idx]['area'] += sec_mask['area']
            else:
                # Se non ci sono candidati, assorbila nello sfondo come fallback
                print(f"[SAM] Maschera secondaria senza candidati, assorbita dallo sfondo (fallback)")
                primary_masks[0]['segmentation'] |= sec_seg
                primary_masks[0]['area'] += sec_mask['area']
    
        return [primary_masks[0]] + target_masks
    
    def _extract_face_and_hair(self, mask_2, mask_3, frame_shape):
        circularity_2 = self._calculate_circularity(mask_2['segmentation'])
        circularity_3 = self._calculate_circularity(mask_3['segmentation'])
        
        if circularity_2 > circularity_3:
            face_mask = mask_2['segmentation']
            hair_mask = mask_3['segmentation']
        else:
            face_mask = mask_3['segmentation']
            hair_mask = mask_2['segmentation']
        
        face_bbox = self._compute_face_bbox(face_mask, frame_shape)
        
        return face_mask, hair_mask, face_bbox
    
    def _build_sam_mask(self, H, W, body_mask, hair_mask):
        sam_result = np.full((H, W), Config.BACKGROUND_CLASS, dtype=np.uint8)
        sam_result[body_mask] = Config.BODY_CLASS
        
        if hair_mask is not None:
            sam_result[hair_mask] = Config.HAIR_CLASS
        
        return sam_result
    
    def _calculate_circularity(self, mask):
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0.0
        
        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        if perimeter == 0:
            return 0.0
        
        return (4 * np.pi * area) / (perimeter ** 2)
    
    def _compute_face_bbox(self, face_mask, frame_shape):
        if not np.any(face_mask):
            return None
    
        H, W = frame_shape
        ys, xs = np.where(face_mask)
        
        x1 = int(xs.min())
        y1 = int(ys.min())
        x2 = int(xs.max())
        y2 = int(ys.max())
        
        margin = 5
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(W, x2 + margin)
        y2 = min(H, y2 + margin)
        
        return (x1, y1, x2, y2)
    
    def clear_cache(self):
        if self.device == "cuda":
            torch.cuda.empty_cache()