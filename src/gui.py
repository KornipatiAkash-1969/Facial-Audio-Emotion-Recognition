"""
Modern Graphical User Interface for Multimodal Emotion Recognition.
Integrates Facial Emotion Recognition, Audio Emotion Recognition, and Multimodal Fusion.
"""

from pathlib import Path
from typing import Optional, Dict, Any
import datetime
import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk

from src.config import (
    EMOTIONS,
    EMOTION_HEX_COLORS,
    FACE_SAMPLES_DIR,
    AUDIO_SAMPLES_DIR,
    FACIAL_MODEL_PATH,
    AUDIO_MODEL_PATH,
    HAAR_CASCADE_PATH,
    get_emoji_path,
    normalize_emotion,
)
from src.facial_detector import FacialEmotionDetector
from src.audio_detector import AudioEmotionDetector
from src.multimodal_fusion import MultimodalEmotionFusion


class MultimodalEmotionApp:
    """Unified Tkinter GUI application for Facial, Audio, and Multimodal Emotion Recognition."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Multimodal Emotion Recognition System")
        self.root.geometry("1100x800")
        self.root.minsize(950, 700)

        # Style and Theme Configuration
        self._setup_styles()

        # Initialize ML Engines
        self.face_detector: Optional[FacialEmotionDetector] = None
        self.audio_detector: Optional[AudioEmotionDetector] = None
        self.fusion_engine = MultimodalEmotionFusion()

        self._init_models()

        # Application State
        self.webcam_cap: Optional[cv2.VideoCapture] = None
        self.is_webcam_running = False
        self.last_webcam_frame: Optional[np.ndarray] = None
        self.current_face_image_path: Optional[str] = None
        self.current_audio_file_path: Optional[str] = None

        # Multimodal Inputs State
        self.mm_face_path: Optional[str] = None
        self.mm_face_frame: Optional[np.ndarray] = None
        self.mm_audio_path: Optional[str] = None

        # Prediction History Cache
        self.history = []

        # Photo references to prevent garbage collection
        self.photo_refs = {}

        # Build UI Components
        self._build_header()
        self._build_notebook()

        # Handle window close cleanly
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _setup_styles(self):
        """Configure ttk styling for a clean, modern aesthetic."""
        self.style = ttk.Style()
        try:
            self.style.theme_use("clam")
        except Exception:
            pass

        # Global fonts and colors
        self.bg_color = "#F4F6F9"
        self.card_bg = "#FFFFFF"
        self.primary_color = "#2980B9"
        self.header_bg = "#1F2937"
        self.header_fg = "#FFFFFF"

        self.root.configure(bg=self.bg_color)

        self.style.configure(".", background=self.bg_color, font=("Segoe UI", 10))
        self.style.configure("TNotebook", background=self.bg_color)
        self.style.configure(
            "TNotebook.Tab",
            font=("Segoe UI", 10, "bold"),
            padding=[16, 8],
            background="#E5E7EB",
        )
        self.style.map(
            "TNotebook.Tab",
            background=[("selected", "#FFFFFF")],
            foreground=[("selected", self.primary_color)],
        )

        self.style.configure(
            "Primary.TButton",
            font=("Segoe UI", 10, "bold"),
            padding=6,
            background=self.primary_color,
            foreground="#FFFFFF",
        )
        self.style.configure(
            "Success.TButton",
            font=("Segoe UI", 10, "bold"),
            padding=6,
            background="#27AE60",
            foreground="#FFFFFF",
        )
        self.style.configure(
            "Danger.TButton",
            font=("Segoe UI", 10, "bold"),
            padding=6,
            background="#C0392B",
            foreground="#FFFFFF",
        )

    def _init_models(self):
        """Load facial and audio models with informative status tracking."""
        try:
            self.face_detector = FacialEmotionDetector()
            self.face_status = "Ready"
        except Exception as err:
            self.face_detector = None
            self.face_status = f"Error: {err}"

        try:
            self.audio_detector = AudioEmotionDetector()
            self.audio_status = "Ready"
        except Exception as err:
            self.audio_detector = None
            self.audio_status = f"Error: {err}"

    def _build_header(self):
        """Create header banner with app title and model readiness pills."""
        header_frame = tk.Frame(self.root, bg=self.header_bg, height=75)
        header_frame.pack(fill=tk.X, side=tk.TOP)

        title_box = tk.Frame(header_frame, bg=self.header_bg)
        title_box.pack(side=tk.LEFT, padx=20, pady=12)

        title_lbl = tk.Label(
            title_box,
            text="Multimodal Emotion Recognition System",
            font=("Segoe UI", 17, "bold"),
            bg=self.header_bg,
            fg=self.header_fg,
        )
        title_lbl.pack(anchor="w")

        sub_lbl = tk.Label(
            title_box,
            text="Visual Facial Expressions • Speech Acoustic Prosody • Multimodal Fusion",
            font=("Segoe UI", 9),
            bg=self.header_bg,
            fg="#9CA3AF",
        )
        sub_lbl.pack(anchor="w")

        # Status Badges
        badge_box = tk.Frame(header_frame, bg=self.header_bg)
        badge_box.pack(side=tk.RIGHT, padx=20, pady=15)

        face_color = "#10B981" if self.face_status == "Ready" else "#EF4444"
        audio_color = "#10B981" if self.audio_status == "Ready" else "#EF4444"

        face_badge = tk.Label(
            badge_box,
            text=f"Face Model: {self.face_status}",
            bg=face_color,
            fg="#FFFFFF",
            font=("Segoe UI", 9, "bold"),
            padx=10,
            pady=4,
            relief="flat",
        )
        face_badge.pack(side=tk.LEFT, padx=5)

        audio_badge = tk.Label(
            badge_box,
            text=f"Audio Model: {self.audio_status}",
            bg=audio_color,
            fg="#FFFFFF",
            font=("Segoe UI", 9, "bold"),
            padx=10,
            pady=4,
            relief="flat",
        )
        audio_badge.pack(side=tk.LEFT, padx=5)

    def _build_notebook(self):
        """Build main tabbed interface."""
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=15, pady=12)

        # Tab 1: Facial Emotion
        self.tab_facial = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_facial, text="  Facial Emotion  ")
        self._build_facial_tab()

        # Tab 2: Audio Emotion
        self.tab_audio = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_audio, text="  Audio Emotion  ")
        self._build_audio_tab()

        # Tab 3: Multimodal Fusion
        self.tab_fusion = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_fusion, text="  Multimodal Fusion  ")
        self._build_fusion_tab()

        # Tab 4: History & Analytics
        self.tab_history = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_history, text="  Prediction History  ")
        self._build_history_tab()

        # Tab 5: About & System Info
        self.tab_about = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_about, text="  About & System  ")
        self._build_about_tab()

    # =========================================================================
    # TAB 1: FACIAL EMOTION RECOGNITION
    # =========================================================================
    def _build_facial_tab(self):
        tab = self.tab_facial

        # Controls Toolbar
        toolbar = tk.Frame(tab, bg=self.card_bg, pady=10, padx=15)
        toolbar.pack(fill=tk.X, side=tk.TOP, pady=(0, 10))

        btn_upload = ttk.Button(
            toolbar,
            text="Upload Image",
            style="Primary.TButton",
            command=self._facial_upload_image,
        )
        btn_upload.pack(side=tk.LEFT, padx=6)

        self.btn_webcam_toggle = ttk.Button(
            toolbar,
            text="Start Webcam",
            style="Success.TButton",
            command=self._facial_toggle_webcam,
        )
        self.btn_webcam_toggle.pack(side=tk.LEFT, padx=6)

        btn_snap = ttk.Button(
            toolbar,
            text="Capture Frame",
            command=self._facial_capture_webcam,
        )
        btn_snap.pack(side=tk.LEFT, padx=6)

        # Sample Faces Dropdown
        tk.Label(toolbar, text="Sample Faces:", bg=self.card_bg, font=("Segoe UI", 9, "bold")).pack(
            side=tk.LEFT, padx=(20, 6)
        )
        self.face_sample_var = tk.StringVar()
        sample_files = []
        if FACE_SAMPLES_DIR.exists():
            sample_files = [f.name for f in FACE_SAMPLES_DIR.glob("*.*") if f.suffix.lower() in [".jpg", ".png", ".jpeg"]]
        self.face_sample_combo = ttk.Combobox(
            toolbar,
            textvariable=self.face_sample_var,
            values=sample_files,
            state="readonly",
            width=26,
        )
        self.face_sample_combo.pack(side=tk.LEFT, padx=4)
        self.face_sample_combo.bind("<<ComboboxSelected>>", self._facial_select_sample)

        # Main Body Split (Left: Preview, Right: Analysis)
        body = tk.Frame(tab, bg=self.bg_color)
        body.pack(fill=tk.BOTH, expand=True)

        # Left Preview Card
        left_card = tk.LabelFrame(
            body,
            text=" Visual Feed / Image Preview ",
            bg=self.card_bg,
            font=("Segoe UI", 11, "bold"),
            padx=10,
            pady=10,
        )
        left_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        self.face_canvas = tk.Label(
            left_card,
            bg="#2C3E50",
            text="Upload an image or start the webcam to begin facial emotion recognition",
            fg="#BDC3C7",
            font=("Segoe UI", 11),
        )
        self.face_canvas.pack(fill=tk.BOTH, expand=True)

        # Right Analysis Card
        right_card = tk.LabelFrame(
            body,
            text=" Facial Emotion Metrics ",
            bg=self.card_bg,
            font=("Segoe UI", 11, "bold"),
            padx=15,
            pady=15,
            width=360,
        )
        right_card.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(0, 0))
        right_card.pack_propagate(False)

        # Primary Emotion Display
        self.face_res_lbl = tk.Label(
            right_card,
            text="Awaiting Input",
            font=("Segoe UI", 18, "bold"),
            bg=self.card_bg,
            fg=self.primary_color,
        )
        self.face_res_lbl.pack(pady=(5, 2))

        self.face_conf_lbl = tk.Label(
            right_card,
            text="Confidence: --%",
            font=("Segoe UI", 11),
            bg=self.card_bg,
            fg="#6B7280",
        )
        self.face_conf_lbl.pack(pady=(0, 10))

        # Emoji Icon
        self.face_emoji_lbl = tk.Label(right_card, bg=self.card_bg)
        self.face_emoji_lbl.pack(pady=5)

        self.face_count_lbl = tk.Label(
            right_card,
            text="Faces Detected: 0",
            font=("Segoe UI", 9, "italic"),
            bg=self.card_bg,
            fg="#4B5563",
        )
        self.face_count_lbl.pack(pady=4)

        # Emotion Probabilities Bars
        bars_frame = tk.Frame(right_card, bg=self.card_bg)
        bars_frame.pack(fill=tk.BOTH, expand=True, pady=(15, 0))

        self.face_prob_bars = {}
        for em in EMOTIONS:
            row = tk.Frame(bars_frame, bg=self.card_bg)
            row.pack(fill=tk.X, pady=3)

            lbl = tk.Label(row, text=f"{em:9s}", font=("Segoe UI", 9, "bold"), bg=self.card_bg, width=8, anchor="w")
            lbl.pack(side=tk.LEFT)

            pbar = ttk.Progressbar(row, orient="horizontal", mode="determinate", length=160)
            pbar.pack(side=tk.LEFT, padx=6)

            pct_lbl = tk.Label(row, text="0.0%", font=("Segoe UI", 9), bg=self.card_bg, width=6, anchor="w")
            pct_lbl.pack(side=tk.LEFT)

            self.face_prob_bars[em] = (pbar, pct_lbl)

    def _facial_upload_image(self):
        """Handle upload of a facial image file."""
        if self.is_webcam_running:
            self._facial_stop_webcam()

        path = filedialog.askopenfilename(
            title="Select Face Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp")],
        )
        if path:
            self._process_and_show_face_image(path)

    def _facial_select_sample(self, event=None):
        """Handle selection from sample face images combobox."""
        name = self.face_sample_var.get()
        if not name:
            return
        if self.is_webcam_running:
            self._facial_stop_webcam()

        path = FACE_SAMPLES_DIR / name
        if path.exists():
            self._process_and_show_face_image(str(path))

    def _process_and_show_face_image(self, path: str):
        """Run facial detection and update UI."""
        if not self.face_detector or not self.face_detector.is_ready:
            messagebox.showerror("Error", "Facial detector model is not ready.")
            return

        self.current_face_image_path = path
        self.mm_face_path = path

        try:
            res = self.face_detector.predict(path, annotate=True)
            annotated_bgr = res["annotated_frame"]
            self._render_face_frame_on_canvas(annotated_bgr)
            self._update_facial_metrics(res)

            # Record in history
            self._add_history(
                modality="Face (Image)",
                emotion=res["primary_emotion"],
                confidence=res["primary_confidence"],
                source=os.path.basename(path),
            )
        except Exception as err:
            messagebox.showerror("Processing Error", f"Failed to analyze image: {err}")

    def _render_face_frame_on_canvas(self, bgr_img: np.ndarray):
        """Resize and display an OpenCV BGR image onto the face_canvas widget."""
        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(rgb)

        # Get canvas current size or default
        max_w = max(400, self.face_canvas.winfo_width() - 10)
        max_h = max(350, self.face_canvas.winfo_height() - 10)

        im.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(im)
        self.photo_refs["face_canvas"] = photo
        self.face_canvas.config(image=photo, text="")

    def _update_facial_metrics(self, res: Dict[str, Any]):
        """Update metrics, emoji, and progress bars on the facial tab."""
        emotion = res["primary_emotion"]
        conf = res["primary_confidence"]
        n_faces = res["num_faces"]

        color = EMOTION_HEX_COLORS.get(emotion, self.primary_color)
        self.face_res_lbl.config(text=emotion, fg=color)
        self.face_conf_lbl.config(text=f"Confidence: {conf * 100:.1f}%")
        self.face_count_lbl.config(text=f"Faces Detected: {n_faces}")

        # Update Emoji
        emoji_path = get_emoji_path(emotion)
        if emoji_path and emoji_path.exists():
            em_img = Image.open(str(emoji_path))
            em_img = em_img.resize((85, 85), Image.Resampling.LANCZOS)
            em_photo = ImageTk.PhotoImage(em_img)
            self.photo_refs["face_emoji"] = em_photo
            self.face_emoji_lbl.config(image=em_photo)
        else:
            self.face_emoji_lbl.config(image="")

        # Update bars
        probs = res["primary_probabilities"]
        for em in EMOTIONS:
            val = probs.get(em, 0.0)
            pbar, pct_lbl = self.face_prob_bars[em]
            pbar["value"] = val * 100
            pct_lbl.config(text=f"{val * 100:.1f}%")

    def _facial_toggle_webcam(self):
        """Toggle live webcam streaming on/off."""
        if self.is_webcam_running:
            self._facial_stop_webcam()
        else:
            self._facial_start_webcam()

    def _facial_start_webcam(self):
        """Start the OpenCV webcam video stream without blocking Tkinter."""
        if not self.face_detector or not self.face_detector.is_ready:
            messagebox.showerror("Error", "Facial model is not loaded.")
            return

        self.webcam_cap = cv2.VideoCapture(0)
        if not self.webcam_cap or not self.webcam_cap.isOpened():
            messagebox.showerror("Camera Error", "Could not open webcam (device index 0).")
            self.webcam_cap = None
            return

        self.is_webcam_running = True
        self.btn_webcam_toggle.config(text="Stop Webcam", style="Danger.TButton")
        self._webcam_loop()

    def _facial_stop_webcam(self):
        """Stop the webcam stream and release hardware."""
        self.is_webcam_running = False
        if self.webcam_cap:
            try:
                self.webcam_cap.release()
            except Exception:
                pass
            self.webcam_cap = None
        self.btn_webcam_toggle.config(text="Start Webcam", style="Success.TButton")

    def _webcam_loop(self):
        """Non-blocking recurring webcam frame grabber using root.after."""
        if not self.is_webcam_running or self.webcam_cap is None:
            return

        ret, frame = self.webcam_cap.read()
        if ret and frame is not None:
            self.last_webcam_frame = frame.copy()
            # Run facial prediction on frame
            try:
                res = self.face_detector.predict(frame, annotate=True)
                annotated = res["annotated_frame"]
                self._render_face_frame_on_canvas(annotated)
                self._update_facial_metrics(res)
            except Exception:
                self._render_face_frame_on_canvas(frame)

        # Schedule next frame in ~33ms (approx 30 FPS)
        if self.is_webcam_running:
            self.root.after(33, self._webcam_loop)

    def _facial_capture_webcam(self):
        """Capture the current webcam frame as the selected face input."""
        if self.last_webcam_frame is not None:
            self.mm_face_frame = self.last_webcam_frame.copy()
            messagebox.showinfo("Captured", "Webcam frame captured and saved for Multimodal Fusion analysis.")
        else:
            messagebox.showwarning("Warning", "No active webcam frame to capture.")

    # =========================================================================
    # TAB 2: AUDIO EMOTION RECOGNITION
    # =========================================================================
    def _build_audio_tab(self):
        tab = self.tab_audio

        # Controls Toolbar
        toolbar = tk.Frame(tab, bg=self.card_bg, pady=10, padx=15)
        toolbar.pack(fill=tk.X, side=tk.TOP, pady=(0, 10))

        btn_upload = ttk.Button(
            toolbar,
            text="Upload Audio (.wav)",
            style="Primary.TButton",
            command=self._audio_upload_file,
        )
        btn_upload.pack(side=tk.LEFT, padx=6)

        btn_record = ttk.Button(
            toolbar,
            text="Record Mic (3s)",
            style="Success.TButton",
            command=self._audio_record_mic,
        )
        btn_record.pack(side=tk.LEFT, padx=6)

        btn_play = ttk.Button(
            toolbar,
            text="Play Audio",
            command=self._audio_play_current,
        )
        btn_play.pack(side=tk.LEFT, padx=6)

        # Sample Audios Dropdown
        tk.Label(toolbar, text="Sample Audios:", bg=self.card_bg, font=("Segoe UI", 9, "bold")).pack(
            side=tk.LEFT, padx=(20, 6)
        )
        self.audio_sample_var = tk.StringVar()
        sample_audios = []
        if AUDIO_SAMPLES_DIR.exists():
            sample_audios = [f.name for f in AUDIO_SAMPLES_DIR.glob("*.wav")]
        self.audio_sample_combo = ttk.Combobox(
            toolbar,
            textvariable=self.audio_sample_var,
            values=sample_audios,
            state="readonly",
            width=26,
        )
        self.audio_sample_combo.pack(side=tk.LEFT, padx=4)
        self.audio_sample_combo.bind("<<ComboboxSelected>>", self._audio_select_sample)

        # Main Body Split (Left: Audio Info Card, Right: Metrics)
        body = tk.Frame(tab, bg=self.bg_color)
        body.pack(fill=tk.BOTH, expand=True)

        # Left Audio Player / Waveform Card
        left_card = tk.LabelFrame(
            body,
            text=" Speech Signal Details ",
            bg=self.card_bg,
            font=("Segoe UI", 11, "bold"),
            padx=20,
            pady=20,
        )
        left_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        self.audio_file_lbl = tk.Label(
            left_card,
            text="No audio file loaded",
            font=("Segoe UI", 12, "bold"),
            bg=self.card_bg,
            fg="#374151",
            anchor="w",
        )
        self.audio_file_lbl.pack(fill=tk.X, pady=(10, 5))

        self.audio_duration_lbl = tk.Label(
            left_card,
            text="Duration: -- sec  |  Format: WAV  |  Features: 40 MFCCs",
            font=("Segoe UI", 10),
            bg=self.card_bg,
            fg="#6B7280",
            anchor="w",
        )
        self.audio_duration_lbl.pack(fill=tk.X, pady=(0, 20))

        # Audio Banner Graphic
        banner_path = Path(__file__).resolve().parent.parent / "assets" / "banners" / "audio_banner.png"
        if banner_path.exists():
            try:
                b_img = Image.open(str(banner_path))
                b_img.thumbnail((450, 150), Image.Resampling.LANCZOS)
                b_photo = ImageTk.PhotoImage(b_img)
                self.photo_refs["audio_banner"] = b_photo
                b_lbl = tk.Label(left_card, image=b_photo, bg=self.card_bg)
                b_lbl.pack(pady=10)
            except Exception:
                pass

        self.audio_status_lbl = tk.Label(
            left_card,
            text="Upload a .wav file or click 'Record Mic' to predict emotion from speech acoustics.",
            font=("Segoe UI", 10, "italic"),
            bg=self.card_bg,
            fg="#4B5563",
            wraplength=400,
        )
        self.audio_status_lbl.pack(fill=tk.X, pady=20)

        # Right Analysis Card
        right_card = tk.LabelFrame(
            body,
            text=" Audio Emotion Metrics ",
            bg=self.card_bg,
            font=("Segoe UI", 11, "bold"),
            padx=15,
            pady=15,
            width=360,
        )
        right_card.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(0, 0))
        right_card.pack_propagate(False)

        # Primary Audio Emotion
        self.audio_res_lbl = tk.Label(
            right_card,
            text="Awaiting Audio",
            font=("Segoe UI", 18, "bold"),
            bg=self.card_bg,
            fg=self.primary_color,
        )
        self.audio_res_lbl.pack(pady=(5, 2))

        self.audio_conf_lbl = tk.Label(
            right_card,
            text="Confidence: --%",
            font=("Segoe UI", 11),
            bg=self.card_bg,
            fg="#6B7280",
        )
        self.audio_conf_lbl.pack(pady=(0, 10))

        # Emoji Icon
        self.audio_emoji_lbl = tk.Label(right_card, bg=self.card_bg)
        self.audio_emoji_lbl.pack(pady=5)

        # Audio Bars
        bars_frame = tk.Frame(right_card, bg=self.card_bg)
        bars_frame.pack(fill=tk.BOTH, expand=True, pady=(15, 0))

        self.audio_prob_bars = {}
        for em in EMOTIONS:
            row = tk.Frame(bars_frame, bg=self.card_bg)
            row.pack(fill=tk.X, pady=3)

            lbl = tk.Label(row, text=f"{em:9s}", font=("Segoe UI", 9, "bold"), bg=self.card_bg, width=8, anchor="w")
            lbl.pack(side=tk.LEFT)

            pbar = ttk.Progressbar(row, orient="horizontal", mode="determinate", length=160)
            pbar.pack(side=tk.LEFT, padx=6)

            pct_lbl = tk.Label(row, text="0.0%", font=("Segoe UI", 9), bg=self.card_bg, width=6, anchor="w")
            pct_lbl.pack(side=tk.LEFT)

            self.audio_prob_bars[em] = (pbar, pct_lbl)

    def _audio_upload_file(self):
        """Upload an audio file (.wav)."""
        path = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=[("Audio WAV files", "*.wav"), ("All audio", "*.wav *.mp3 *.flac *.ogg")],
        )
        if path:
            self._process_and_show_audio(path)

    def _audio_select_sample(self, event=None):
        """Select a sample audio file."""
        name = self.audio_sample_var.get()
        if not name:
            return
        path = AUDIO_SAMPLES_DIR / name
        if path.exists():
            self._process_and_show_audio(str(path))

    def _process_and_show_audio(self, path: str):
        """Run speech emotion inference on selected audio file."""
        if not self.audio_detector or not self.audio_detector.is_ready:
            messagebox.showerror("Error", "Audio detector model is not ready.")
            return

        self.current_audio_file_path = path
        self.mm_audio_path = path

        try:
            res = self.audio_detector.predict(path)
            self.audio_file_lbl.config(text=os.path.basename(path))
            self.audio_duration_lbl.config(
                text=f"Duration: {res['duration_seconds']} sec  |  Format: WAV  |  Features: 40 MFCCs"
            )
            self.audio_status_lbl.config(text="Audio analysis completed successfully.")
            self._update_audio_metrics(res)

            self._add_history(
                modality="Audio",
                emotion=res["emotion"],
                confidence=res["confidence"],
                source=os.path.basename(path),
            )
        except Exception as err:
            messagebox.showerror("Audio Error", f"Failed to analyze audio: {err}")

    def _audio_record_mic(self):
        """Record a 3-second audio clip from the microphone."""
        if not self.audio_detector:
            return
        try:
            self.audio_status_lbl.config(text="Recording audio from microphone... Speak now!")
            self.root.update()

            rec_dir = Path(__file__).resolve().parent.parent / "assets" / "samples" / "audio"
            rec_dir.mkdir(parents=True, exist_ok=True)
            rec_path = rec_dir / "mic_recording.wav"

            self.audio_detector.record_microphone(duration=3.0, save_path=rec_path)
            self._process_and_show_audio(str(rec_path))
        except Exception as err:
            messagebox.showerror("Recording Error", f"Could not record audio: {err}")
            self.audio_status_lbl.config(text="Microphone recording failed.")

    def _audio_play_current(self):
        """Play current audio file."""
        if self.current_audio_file_path and os.path.exists(self.current_audio_file_path):
            AudioEmotionDetector.play_audio(self.current_audio_file_path)
        else:
            messagebox.showwarning("Warning", "No audio file loaded to play.")

    def _update_audio_metrics(self, res: Dict[str, Any]):
        """Update audio emotion displays and progress bars."""
        emotion = res["emotion"]
        conf = res["confidence"]

        color = EMOTION_HEX_COLORS.get(emotion, self.primary_color)
        self.audio_res_lbl.config(text=emotion, fg=color)
        self.audio_conf_lbl.config(text=f"Confidence: {conf * 100:.1f}%")

        emoji_path = get_emoji_path(emotion)
        if emoji_path and emoji_path.exists():
            em_img = Image.open(str(emoji_path))
            em_img = em_img.resize((85, 85), Image.Resampling.LANCZOS)
            em_photo = ImageTk.PhotoImage(em_img)
            self.photo_refs["audio_emoji"] = em_photo
            self.audio_emoji_lbl.config(image=em_photo)
        else:
            self.audio_emoji_lbl.config(image="")

        probs = res["probabilities"]
        for em in EMOTIONS:
            val = probs.get(em, 0.0)
            pbar, pct_lbl = self.audio_prob_bars[em]
            pbar["value"] = val * 100
            pct_lbl.config(text=f"{val * 100:.1f}%")

    # =========================================================================
    # TAB 3: MULTIMODAL FUSION
    # =========================================================================
    def _build_fusion_tab(self):
        tab = self.tab_fusion

        # Top Inputs Configuration Card
        cfg_card = tk.LabelFrame(
            tab,
            text=" Multimodal Inputs & Fusion Parameters ",
            bg=self.card_bg,
            font=("Segoe UI", 11, "bold"),
            padx=15,
            pady=12,
        )
        cfg_card.pack(fill=tk.X, side=tk.TOP, pady=(0, 10))

        # Inputs Selection Row
        input_row = tk.Frame(cfg_card, bg=self.card_bg)
        input_row.pack(fill=tk.X, pady=4)

        # Face Input
        btn_pick_face = ttk.Button(
            input_row,
            text="Select Face Image",
            command=self._mm_pick_face,
        )
        btn_pick_face.pack(side=tk.LEFT, padx=5)

        self.mm_face_lbl = tk.Label(
            input_row,
            text="No face selected",
            font=("Segoe UI", 9, "italic"),
            bg=self.card_bg,
            fg="#4B5563",
            width=26,
            anchor="w",
        )
        self.mm_face_lbl.pack(side=tk.LEFT, padx=5)

        # Audio Input
        btn_pick_audio = ttk.Button(
            input_row,
            text="Select Audio (.wav)",
            command=self._mm_pick_audio,
        )
        btn_pick_audio.pack(side=tk.LEFT, padx=(20, 5))

        self.mm_audio_lbl = tk.Label(
            input_row,
            text="No audio selected",
            font=("Segoe UI", 9, "italic"),
            bg=self.card_bg,
            fg="#4B5563",
            width=26,
            anchor="w",
        )
        self.mm_audio_lbl.pack(side=tk.LEFT, padx=5)

        # Weight Controls & Execute Button
        ctrl_row = tk.Frame(cfg_card, bg=self.card_bg)
        ctrl_row.pack(fill=tk.X, pady=(12, 4))

        tk.Label(
            ctrl_row,
            text="Face Weight:",
            font=("Segoe UI", 9, "bold"),
            bg=self.card_bg,
        ).pack(side=tk.LEFT, padx=(5, 5))

        self.face_weight_var = tk.DoubleVar(value=0.5)
        self.weight_slider = ttk.Scale(
            ctrl_row,
            from_=0.0,
            to=1.0,
            variable=self.face_weight_var,
            orient="horizontal",
            length=180,
            command=self._mm_on_weight_change,
        )
        self.weight_slider.pack(side=tk.LEFT, padx=5)

        self.weight_val_lbl = tk.Label(
            ctrl_row,
            text="Face: 50% | Voice: 50%",
            font=("Segoe UI", 9),
            bg=self.card_bg,
            width=22,
        )
        self.weight_val_lbl.pack(side=tk.LEFT, padx=5)

        btn_run_fusion = ttk.Button(
            ctrl_row,
            text="Analyze Multimodal Fusion",
            style="Primary.TButton",
            command=self._run_multimodal_fusion,
        )
        btn_run_fusion.pack(side=tk.RIGHT, padx=10)

        # Bottom Results Dashboard
        res_card = tk.LabelFrame(
            tab,
            text=" Affective Computing Fusion Dashboard ",
            bg=self.card_bg,
            font=("Segoe UI", 11, "bold"),
            padx=15,
            pady=12,
        )
        res_card.pack(fill=tk.BOTH, expand=True)

        # Top summary card (Fused Emotion + Congruency Badge)
        sum_row = tk.Frame(res_card, bg=self.card_bg)
        sum_row.pack(fill=tk.X, pady=5)

        self.mm_fused_res_lbl = tk.Label(
            sum_row,
            text="Awaiting Multimodal Inputs",
            font=("Segoe UI", 20, "bold"),
            bg=self.card_bg,
            fg=self.primary_color,
        )
        self.mm_fused_res_lbl.pack(side=tk.LEFT, padx=10)

        self.mm_congruency_badge = tk.Label(
            sum_row,
            text="[ Status: Pending ]",
            font=("Segoe UI", 10, "bold"),
            bg="#9CA3AF",
            fg="#FFFFFF",
            padx=12,
            pady=4,
        )
        self.mm_congruency_badge.pack(side=tk.RIGHT, padx=10)

        # Dual Modality Sub-Cards & Emoji
        cards_row = tk.Frame(res_card, bg=self.card_bg)
        cards_row.pack(fill=tk.X, pady=10)

        # Face Card
        self.mm_face_card_lbl = tk.Label(
            cards_row,
            text="Face Cue: --",
            font=("Segoe UI", 11, "bold"),
            bg="#EFF6FF",
            fg="#1E40AF",
            padx=15,
            pady=10,
            relief="solid",
            bd=1,
            width=28,
        )
        self.mm_face_card_lbl.pack(side=tk.LEFT, padx=5)

        # Big Emoji
        self.mm_emoji_lbl = tk.Label(cards_row, bg=self.card_bg)
        self.mm_emoji_lbl.pack(side=tk.LEFT, expand=True)

        # Voice Card
        self.mm_audio_card_lbl = tk.Label(
            cards_row,
            text="Voice Cue: --",
            font=("Segoe UI", 11, "bold"),
            bg="#F0FDF4",
            fg="#166534",
            padx=15,
            pady=10,
            relief="solid",
            bd=1,
            width=28,
        )
        self.mm_audio_card_lbl.pack(side=tk.RIGHT, padx=5)

        # Fused Probabilities Bars
        bars_frame = tk.Frame(res_card, bg=self.card_bg)
        bars_frame.pack(fill=tk.X, pady=8)

        self.mm_prob_bars = {}
        for em in EMOTIONS:
            row = tk.Frame(bars_frame, bg=self.card_bg)
            row.pack(fill=tk.X, pady=2)

            lbl = tk.Label(row, text=f"{em:9s}", font=("Segoe UI", 9, "bold"), bg=self.card_bg, width=8, anchor="w")
            lbl.pack(side=tk.LEFT)

            pbar = ttk.Progressbar(row, orient="horizontal", mode="determinate", length=280)
            pbar.pack(side=tk.LEFT, padx=8)

            pct_lbl = tk.Label(row, text="0.0%", font=("Segoe UI", 9), bg=self.card_bg, width=6, anchor="w")
            pct_lbl.pack(side=tk.LEFT)

            self.mm_prob_bars[em] = (pbar, pct_lbl)

        # Affective Insight Text Box
        insight_frame = tk.LabelFrame(
            res_card,
            text=" Affective Computing & Behavioral Interpretation ",
            bg=self.card_bg,
            font=("Segoe UI", 10, "bold"),
            padx=12,
            pady=8,
        )
        insight_frame.pack(fill=tk.BOTH, expand=True, pady=(8, 0))

        self.mm_insight_lbl = tk.Label(
            insight_frame,
            text="Select both a face image and a voice audio sample, then click 'Analyze Multimodal Fusion' to generate integrated affective computing results.",
            font=("Segoe UI", 10),
            bg=self.card_bg,
            fg="#374151",
            justify="left",
            wraplength=850,
            anchor="w",
        )
        self.mm_insight_lbl.pack(fill=tk.BOTH, expand=True)

    def _mm_on_weight_change(self, val):
        """Update slider weight percentage label."""
        fw = self.face_weight_var.get()
        aw = 1.0 - fw
        self.weight_val_lbl.config(text=f"Face: {fw * 100:.0f}% | Voice: {aw * 100:.0f}%")

    def _mm_pick_face(self):
        """Pick face image for multimodal analysis."""
        path = filedialog.askopenfilename(
            title="Select Face Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")],
        )
        if path:
            self.mm_face_path = path
            self.mm_face_frame = None
            self.mm_face_lbl.config(text=os.path.basename(path))

    def _mm_pick_audio(self):
        """Pick audio file for multimodal analysis."""
        path = filedialog.askopenfilename(
            title="Select Speech Audio File",
            filetypes=[("Audio WAV files", "*.wav")],
        )
        if path:
            self.mm_audio_path = path
            self.mm_audio_lbl.config(text=os.path.basename(path))

    def _run_multimodal_fusion(self):
        """Execute end-to-end multimodal fusion."""
        # Check inputs
        face_input = self.mm_face_frame if self.mm_face_frame is not None else self.mm_face_path
        if face_input is None and self.current_face_image_path:
            face_input = self.current_face_image_path

        audio_input = self.mm_audio_path
        if audio_input is None and self.current_audio_file_path:
            audio_input = self.current_audio_file_path

        if face_input is None:
            messagebox.showwarning("Missing Input", "Please select or capture a face image first.")
            return
        if audio_input is None:
            messagebox.showwarning("Missing Input", "Please select a speech audio file first.")
            return

        try:
            # 1. Predict Face
            face_res = self.face_detector.predict(face_input)
            # 2. Predict Audio
            audio_res = self.audio_detector.predict(audio_input)

            # 3. Fuse
            fw = self.face_weight_var.get()
            aw = 1.0 - fw
            fusion_res = self.fusion_engine.fuse(
                face_result=face_res,
                audio_result=audio_res,
                face_weight=fw,
                audio_weight=aw,
            )

            # Update UI
            fused_em = fusion_res["fused_emotion"]
            fused_conf = fusion_res["fused_confidence"]
            color = EMOTION_HEX_COLORS.get(fused_em, self.primary_color)

            self.mm_fused_res_lbl.config(
                text=f"{fused_em} ({fused_conf * 100:.1f}%)",
                fg=color,
            )

            # Update Congruency Badge
            if fusion_res["is_congruent"]:
                self.mm_congruency_badge.config(
                    text="✓ CONGRUENT (Synergistic Match)",
                    bg="#10B981",
                )
            else:
                self.mm_congruency_badge.config(
                    text="⚠ INCONGRUENT (Affective Conflict)",
                    bg="#F59E0B",
                )

            # Update sub-cards
            fc = fusion_res["face"]
            ac = fusion_res["audio"]
            self.mm_face_card_lbl.config(
                text=f"Visual: {fc['emotion']} ({fc['confidence']*100:.1f}%)\nWeight: {fc['weight']*100:.0f}%"
            )
            self.mm_audio_card_lbl.config(
                text=f"Acoustic: {ac['emotion']} ({ac['confidence']*100:.1f}%)\nWeight: {ac['weight']*100:.0f}%"
            )

            # Update Big Emoji
            emoji_path = get_emoji_path(fused_em)
            if emoji_path and emoji_path.exists():
                em_img = Image.open(str(emoji_path))
                em_img = em_img.resize((90, 90), Image.Resampling.LANCZOS)
                em_photo = ImageTk.PhotoImage(em_img)
                self.photo_refs["mm_emoji"] = em_photo
                self.mm_emoji_lbl.config(image=em_photo)

            # Update Progress Bars
            c_probs = fusion_res["combined_probabilities"]
            for em in EMOTIONS:
                val = c_probs.get(em, 0.0)
                pbar, pct_lbl = self.mm_prob_bars[em]
                pbar["value"] = val * 100
                pct_lbl.config(text=f"{val * 100:.1f}%")

            # Update Insight text
            self.mm_insight_lbl.config(text=fusion_res["insight"])

            # Record in history
            self._add_history(
                modality="Multimodal",
                emotion=fused_em,
                confidence=fused_conf,
                source=f"Face: {fc['emotion']} | Audio: {ac['emotion']}",
            )

        except Exception as err:
            messagebox.showerror("Fusion Error", f"Failed to execute multimodal fusion: {err}")

    # =========================================================================
    # TAB 4: PREDICTION HISTORY & ANALYTICS
    # =========================================================================
    def _build_history_tab(self):
        tab = self.tab_history

        toolbar = tk.Frame(tab, bg=self.card_bg, pady=8, padx=15)
        toolbar.pack(fill=tk.X, side=tk.TOP, pady=(0, 10))

        btn_clear = ttk.Button(toolbar, text="Clear History", command=self._history_clear)
        btn_clear.pack(side=tk.LEFT, padx=6)

        btn_export = ttk.Button(toolbar, text="Export CSV", command=self._history_export)
        btn_export.pack(side=tk.LEFT, padx=6)

        # Scrollable Treeview Table
        table_frame = tk.Frame(tab, bg=self.card_bg)
        table_frame.pack(fill=tk.BOTH, expand=True)

        cols = ("idx", "time", "modality", "emotion", "confidence", "source")
        self.tree = ttk.Treeview(table_frame, columns=cols, show="headings")

        self.tree.heading("idx", text="#")
        self.tree.heading("time", text="Timestamp")
        self.tree.heading("modality", text="Modality")
        self.tree.heading("emotion", text="Predicted Emotion")
        self.tree.heading("confidence", text="Confidence")
        self.tree.heading("source", text="Source / Cues")

        self.tree.column("idx", width=40, anchor="center")
        self.tree.column("time", width=140, anchor="center")
        self.tree.column("modality", width=120, anchor="center")
        self.tree.column("emotion", width=130, anchor="center")
        self.tree.column("confidence", width=100, anchor="center")
        self.tree.column("source", width=380, anchor="w")

        scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)

        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def _add_history(self, modality: str, emotion: str, confidence: float, source: str):
        """Append a prediction entry to history."""
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        idx = len(self.history) + 1
        entry = (idx, ts, modality, emotion, f"{confidence * 100:.1f}%", source)
        self.history.append(entry)
        self.tree.insert("", tk.END, values=entry)

    def _history_clear(self):
        """Clear all prediction history."""
        self.history.clear()
        for item in self.tree.get_children():
            self.tree.delete(item)

    def _history_export(self):
        """Export prediction history to a CSV file."""
        if not self.history:
            messagebox.showwarning("Empty", "No prediction history to export.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
        )
        if path:
            try:
                import csv
                with open(path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(["Index", "Timestamp", "Modality", "Emotion", "Confidence", "Source"])
                    for row in self.history:
                        writer.writerow(row)
                messagebox.showinfo("Exported", f"Successfully exported history to {path}")
            except Exception as err:
                messagebox.showerror("Error", f"Failed to export CSV: {err}")

    # =========================================================================
    # TAB 5: ABOUT & SYSTEM INFO
    # =========================================================================
    def _build_about_tab(self):
        tab = self.tab_about

        card = tk.LabelFrame(
            tab,
            text=" System Architecture & Dataset Credits ",
            bg=self.card_bg,
            font=("Segoe UI", 11, "bold"),
            padx=20,
            pady=20,
        )
        card.pack(fill=tk.BOTH, expand=True)

        info_text = (
            "MULTIMODAL EMOTION RECOGNITION SYSTEM\n"
            "======================================\n\n"
            "This software combines state-of-the-art Computer Vision and Speech Signal Processing\n"
            "to recognize human affective states across 7 universal emotions:\n"
            "  • Angry  • Disgust  • Fear  • Happy  • Neutral  • Sad  • Surprise\n\n"
            "MODALITIES & MODELS:\n"
            "1. Facial Emotion Recognition:\n"
            f"   - Cascade: Haar Frontal Face Default ({HAAR_CASCADE_PATH.name})\n"
            f"   - Deep Neural Network: 4-Block Convolutional Neural Network ({FACIAL_MODEL_PATH.name})\n"
            "   - Input: 48x48 Grayscale Face Region\n\n"
            "2. Speech Emotion Recognition:\n"
            f"   - Classifier: Multi-Layer Perceptron / Scikit-Learn ({AUDIO_MODEL_PATH.name})\n"
            "   - Feature Extraction: 40 Mel-Frequency Cepstral Coefficients (MFCCs)\n"
            "   - Dataset: Toronto Emotional Speech Set (TESS)\n\n"
            "3. Multimodal Decision-Level Fusion:\n"
            "   - Weighted Probability Integration with configurable bias sliders\n"
            "   - Cross-Modal Emotional Congruency & Affective Dissonance Detection\n\n"
            "ENVIRONMENT SPECIFICATIONS:\n"
            f"   - Python: {sys.version.split()[0]}\n"
            f"   - OpenCV: {cv2.__version__}\n"
            "   - Frameworks: TensorFlow / Keras, Librosa, SoundFile, Scikit-Learn\n"
        )

        txt_box = tk.Text(
            card,
            font=("Consolas", 10),
            bg="#F9FAFB",
            fg="#1F2937",
            relief="solid",
            bd=1,
            padx=15,
            pady=15,
        )
        txt_box.insert(tk.END, info_text)
        txt_box.config(state="disabled")
        txt_box.pack(fill=tk.BOTH, expand=True)

    def on_close(self):
        """Cleanly release resources on application exit."""
        self._facial_stop_webcam()
        self.root.destroy()


def launch_app():
    """Launch the Tkinter application."""
    root = tk.Tk()
    app = MultimodalEmotionApp(root)
    root.mainloop()


if __name__ == "__main__":
    launch_app()
