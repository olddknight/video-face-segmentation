import os
from PyQt5.QtWidgets import (QMainWindow, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QProgressBar, QFileDialog, 
                             QWidget, QMessageBox, QTextEdit, QCheckBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

class MainWindow(QMainWindow):
    def __init__(self, processing_thread_class):
        super().__init__()
        self.processing_thread_class = processing_thread_class
        self.video_path = None
        self.processing_thread = None
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle("Video Face Analysis")
        self.setGeometry(100, 100, 900, 700)
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()
        
        title = QLabel("Video Face Analysis")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        video_layout = QHBoxLayout()
        self.select_btn = QPushButton("Seleziona Video")
        self.select_btn.clicked.connect(self.select_video)
        self.video_label = QLabel("Nessun video selezionato")
        video_layout.addWidget(self.select_btn)
        video_layout.addWidget(self.video_label)
        layout.addLayout(video_layout)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        self.progress_label = QLabel("")
        self.progress_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.progress_label)
        
        self.time_label = QLabel("")
        self.time_label.setAlignment(Qt.AlignCenter)
        self.time_label.setStyleSheet("color: #666; font-weight: bold;")
        layout.addWidget(self.time_label)
        
        button_layout = QHBoxLayout()
        self.process_btn = QPushButton("Avvia Analisi")
        self.process_btn.clicked.connect(self.start_processing)
        self.process_btn.setEnabled(False)
        
        self.predictor_checkbox = QCheckBox("Con Predictor")
        self.predictor_checkbox.setToolTip(
            "Abilita modalità Generator + Predictor per elaborazione ottimizzata.\n"
            "Disabilitato: usa solo Generator per ogni frame (più lento, più preciso).\n"
            "Abilitato: Generator ogni 10 frame, Predictor per frame intermedi (più veloce)."
        )
        self.predictor_checkbox.setChecked(False)
        
        self.stop_btn = QPushButton("Ferma")
        self.stop_btn.clicked.connect(self.stop_processing)
        self.stop_btn.setEnabled(False)
        button_layout.addWidget(self.process_btn)
        button_layout.addWidget(self.predictor_checkbox)
        button_layout.addWidget(self.stop_btn)
        layout.addLayout(button_layout)
        
        log_label = QLabel("Log:")
        log_label.setFont(QFont("Arial", 10, QFont.Bold))
        layout.addWidget(log_label)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)
        
        central_widget.setLayout(layout)
    
    def select_video(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Seleziona Video", "", "Video Files (*.mp4 *.avi *.mov *.mkv)"
        )
        if file_path:
            self.video_path = file_path
            self.video_label.setText(os.path.basename(file_path))
            self.process_btn.setEnabled(True)
    
    def start_processing(self):
        if not self.video_path:
            QMessageBox.warning(self, "Errore", "Seleziona un video")
            return
        
        video_name = os.path.splitext(os.path.basename(self.video_path))[0]
        output_video_path = f"outputs/{video_name}_segmented.mp4"
        os.makedirs("outputs", exist_ok=True)
        
        self.select_btn.setEnabled(False)
        self.process_btn.setEnabled(False)
        self.predictor_checkbox.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.log_text.clear()
        self.time_label.setText("")
        
        use_predictor = self.predictor_checkbox.isChecked()
        
        self.processing_thread = self.processing_thread_class(
            self.video_path, 
            output_video_path,
            use_predictor=use_predictor
        )
        self.processing_thread.progress_updated.connect(self.update_progress)
        self.processing_thread.finished_processing.connect(self.processing_finished)
        self.processing_thread.log_message.connect(self.add_log_message)
        self.processing_thread.start()
        
        mode_text = "Generator + Predictor" if use_predictor else "Solo Generator"
        self.add_log_message(f"Modalità: {mode_text}")
    
    def stop_processing(self):
        if self.processing_thread and self.processing_thread.isRunning():
            self.processing_thread.stop_processing()
            self.stop_btn.setEnabled(False)
    
    def update_progress(self, current, total, message, time_remaining):
        progress = int((current / total) * 100) if total > 0 else 0
        self.progress_bar.setValue(progress)
        self.progress_label.setText(message)
        self.time_label.setText(f"Tempo residuo: {time_remaining}")
    
    def add_log_message(self, message):
        self.log_text.append(message)
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )
    
    def processing_finished(self, success, message):
        self.select_btn.setEnabled(True)
        self.process_btn.setEnabled(True)
        self.predictor_checkbox.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.time_label.setText("")
        
        if success:
            QMessageBox.information(self, "Completato ✓", message)
            self.progress_label.setText("Completato ✓")
            self.progress_label.setStyleSheet("color: green; font-weight: bold;")
        else:
            QMessageBox.critical(self, "Errore", message)
            self.progress_label.setText("Errore ✗")
            self.progress_label.setStyleSheet("color: red; font-weight: bold;")