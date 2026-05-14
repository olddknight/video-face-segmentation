import os
import numpy as np
import shutil
import gc
from config import Config

class MaskSaver:
    """
    Salva le maschere di segmentazione finali per ogni frame del video.
    Le maschere vengono salvate immediatamente su disco in formato .npz individuale,
    poi consolidate in un unico file alla fine.
    """
    
    def __init__(self, video_name):
        """
        Inizializza il MaskSaver.
        
        Args:
            video_name: Nome del video (senza estensione) per identificare la cartella di output
        """
        self.video_name = video_name
        self.temp_dir = os.path.join(Config.OUTPUT_DIR, "masks_data", f"{video_name}_temp")
        self.output_path = os.path.join(Config.OUTPUT_DIR, "masks_data", f"{video_name}_masks.npz")
        os.makedirs(self.temp_dir, exist_ok=True)
        self.frame_count = 0
    
    def add_frame_mask(self, frame_number, mask):
        """
        Salva immediatamente la maschera di un frame su disco (temporaneo).
        
        Args:
            frame_number: Numero del frame (int)
            mask: Maschera di segmentazione come numpy array (H x W)
        """
        if mask is None:
            print(f"[MaskSaver] Warning: Maschera None per frame {frame_number}")
            return
        
        mask_uint8 = mask.astype(np.uint8)
        
        # Salva temporaneamente
        temp_path = os.path.join(self.temp_dir, f"frame_{frame_number:04d}.npz")
        np.savez_compressed(temp_path, mask=mask_uint8)
        
        self.frame_count += 1
        
        # Log periodico ogni 50 frame
        if self.frame_count % 50 == 0:
            print(f"[MaskSaver] Salvate {self.frame_count} maschere (temporanee)")
    
    def save(self):
        """
        Consolida con streaming (NO RAM OVERFLOW)
        """
        if self.frame_count == 0:
            return None
        
        print(f"[MaskSaver] Consolidamento streaming di {self.frame_count} maschere...")
        
        temp_files = sorted([f for f in os.listdir(self.temp_dir) if f.endswith('.npz')])
        
        import zipfile
        import io
        
        with zipfile.ZipFile(self.output_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
            for i, temp_file in enumerate(temp_files):
                frame_num = int(temp_file.split('_')[1].split('.')[0])
                temp_path = os.path.join(self.temp_dir, temp_file)
                
                # Carica UNA SOLA maschera
                data = np.load(temp_path)
                mask = data['mask']
                
                # Serializza in buffer
                buffer = io.BytesIO()
                np.save(buffer, mask)
                buffer.seek(0)
                
                # Scrivi nel ZIP
                key = f"frame_{frame_num:04d}.npy"
                zf.writestr(key, buffer.read())
                
                data.close()
                
                if (i + 1) % 100 == 0:
                    print(f"  Scritti {i + 1}/{len(temp_files)} frame...")
            
        # Ora elimina i file temporanei (con retry su Windows)
        print(f"[MaskSaver] Pulizia file temporanei...")
        self._safe_rmtree(self.temp_dir)
        
        print(f"[MaskSaver] ✓ Salvate {self.frame_count} maschere in {self.output_path}")
        print(f"[MaskSaver] ✓ Dimensione finale: {os.path.getsize(self.output_path) / (1024 * 1024):.2f} MB")
        
        return self.output_path
    
    def _safe_rmtree(self, path, max_retries=3):
        """
        Elimina una directory con retry per gestire file bloccati su Windows.
        
        Args:
            path: Path della directory da eliminare
            max_retries: Numero massimo di tentativi
        """
        import time
        
        for attempt in range(max_retries):
            try:
                shutil.rmtree(path)
                return  # Successo
            except PermissionError as e:
                if attempt < max_retries - 1:
                    print(f"[MaskSaver] Tentativo {attempt + 1}/{max_retries} fallito, attesa 1s...")
                    time.sleep(1)
                    gc.collect()  # Forza garbage collection
                else:
                    print(f"[MaskSaver] Warning: Impossibile eliminare {path}: {e}")
                    print(f"[MaskSaver] I file temporanei possono essere eliminati manualmente")
    
    def get_frame_count(self):
        """Restituisce il numero di maschere salvate."""
        return self.frame_count
    
    @staticmethod
    def load_masks(masks_path):
        """
        Carica tutte le maschere da un file .npz consolidato.
        
        Args:
            masks_path: Path del file .npz
            
        Returns:
            dict: Dizionario {frame_number: numpy_array}
        """
        if not os.path.exists(masks_path):
            raise FileNotFoundError(f"File non trovato: {masks_path}")
        
        data = np.load(masks_path)
        
        try:
            masks_dict = {}
            for key in data.keys():
                frame_num = int(key.split('_')[1])
                masks_dict[frame_num] = data[key].copy()  # Copia per sicurezza
            
            print(f"[MaskSaver] Caricate {len(masks_dict)} maschere da {masks_path}")
            return masks_dict
        
        finally:
            data.close()  # Chiudi esplicitamente
    
    @staticmethod
    def load_single_mask(masks_path, frame_number):
        """
        Carica una singola maschera da un file .npz consolidato.
        
        Args:
            masks_path: Path del file .npz
            frame_number: Numero del frame da caricare
            
        Returns:
            numpy_array: La maschera richiesta
        """
        if not os.path.exists(masks_path):
            raise FileNotFoundError(f"File non trovato: {masks_path}")
        
        data = np.load(masks_path)
        
        try:
            key = f"frame_{frame_number:04d}"
            
            if key not in data:
                raise KeyError(f"Frame {frame_number} non trovato nel file")
            
            return data[key].copy()  # Copia per sicurezza
        
        finally:
            data.close()  # Chiudi esplicitamente
    
    @staticmethod
    def print_masks_info(masks_path):
        """
        Stampa informazioni sulle maschere contenute in un file .npz.
        
        Args:
            masks_path: Path del file .npz
        """
        if not os.path.exists(masks_path):
            print(f"File non trovato: {masks_path}")
            return
        
        data = np.load(masks_path)
        
        try:
            print(f"\n{'='*60}")
            print(f"Informazioni maschere: {os.path.basename(masks_path)}")
            print(f"{'='*60}")
            print(f"Numero totale frame: {len(data.keys())}")
            
            # Estrai numeri frame
            frame_numbers = sorted([int(k.split('_')[1]) for k in data.keys()])
            
            if frame_numbers:
                print(f"Range frame: {frame_numbers[0]} - {frame_numbers[-1]}")
                
                # Info prima maschera
                first_mask = data[f"frame_{frame_numbers[0]:04d}"]
                print(f"Dimensioni maschera: {first_mask.shape}")
                print(f"Tipo dato: {first_mask.dtype}")
                
                # Analizza classi presenti (campiona)
                all_classes = set()
                sample_step = max(1, len(frame_numbers) // 10)
                
                for frame_num in frame_numbers[::sample_step]:
                    mask = data[f"frame_{frame_num:04d}"]
                    all_classes.update(np.unique(mask))
                
                print(f"Classi presenti (campionamento): {sorted(all_classes)}")
            
            # Dimensione file
            file_size_mb = os.path.getsize(masks_path) / (1024 * 1024)
            print(f"Dimensione file: {file_size_mb:.2f} MB")
            
            print(f"{'='*60}\n")
        
        finally:
            data.close()  # Chiudi esplicitamente
