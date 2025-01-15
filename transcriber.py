import tkinter as tk
from tkinter import messagebox, ttk
from faster_whisper import WhisperModel
import pyaudio
import numpy as np
import threading
import queue
import os , sys
import time
import statistics
from transformers import pipeline
from textblob import TextBlob

class AdvancedRealtimeTranscriber:
    def __init__(self, model_size="small", device="cpu", compute_type="int8"):
        self.root = tk.Tk()
        self.root.title("AI Speech Detector")
        self.root.geometry("1200x800")
        
        # Check for microphone before proceeding
        if not self.check_microphone():
            messagebox.showerror("Error", "No microphone detected! Please connect a microphone and restart the application.")
            self.root.destroy()
            return

        # Show loading screen while initializing
        self.show_loading_screen()
        
        # Initialize in background thread
        threading.Thread(target=self.initialize_backend, args=(model_size, device, compute_type)).start()

    def show_loading_screen(self):
        """Show loading progress while initializing"""
        self.loading_frame = tk.Frame(self.root)
        self.loading_frame.place(relx=0.5, rely=0.5, anchor='center')
        
        loading_label = tk.Label(self.loading_frame, text="Initializing...\nPlease wait while we set up the AI models", 
                               font=("Arial", 14))
        loading_label.pack(pady=10)
        
        self.progress = ttk.Progressbar(self.loading_frame, length=300, mode='indeterminate')
        self.progress.pack(pady=10)
        self.progress.start()

    def check_microphone(self):
        """Check if any microphone is available"""
        try:
            p = pyaudio.PyAudio()
            input_devices = []
            for i in range(p.get_device_count()):
                device_info = p.get_device_info_by_index(i)
                if device_info.get('maxInputChannels') > 0:
                    input_devices.append(device_info)
            p.terminate()
            return len(input_devices) > 0
        except Exception:
            return False

    def initialize_backend(self, model_size, device, compute_type):
        """Initialize all backend components"""
        try:
            # Initialize Whisper Model
            self.model = WhisperModel(
                model_size, 
                device=device, 
                compute_type=compute_type,
                cpu_threads=min(os.cpu_count(), 8)
            )

            # AI Detection Markers
            self.ai_markers = {
                'formal_phrases': [
                    'moreover', 'furthermore', 'consequently', 
                    'in conclusion', 'it is worth noting'
                ],
                'ai_disclaimers': [
                    'as an ai', 'i want to be clear', 
                    'it is important to note'
                ],
                'generic_intros': [
                    'i am passionate about', 'i possess',
                    'i am highly', 'my ability to'
                ]
            }

            # Initialize AI Detectors
            self.ai_detectors = self.load_ai_detectors()

            # Audio Settings
            self.CHUNK = 1600
            self.RATE = 16000
            self.CHANNELS = 1

            # Processing Variables
            self.audio_queue = queue.Queue()
            self.stop_event = threading.Event()
            self.text_buffer = []
            self.MIN_CHUNK_WORDS = 10
            self.MAX_CHUNK_WORDS = 100
            self.CHUNK_TIMEOUT = 30
            self.last_chunk_time = time.time()
            self.is_recording = False

            # Initialize PyAudio
            self.p = pyaudio.PyAudio()
            
            # Setup UI after initialization
            self.root.after(0, self.setup_main_ui)
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Initialization Error", 
                f"Error initializing application: {str(e)}\n\nPlease restart the application."))
            self.root.after(0, self.root.destroy)

    def setup_main_ui(self):
        """Setup the main UI after initialization"""
        # Remove loading screen
        self.loading_frame.destroy()

        # Create main frame
        main_frame = tk.Frame(self.root)
        main_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        # Control Panel
        control_frame = tk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))

        self.start_button = tk.Button(
            control_frame, 
            text="Start Recording", 
            command=self.toggle_recording,
            bg="green",
            fg="white",
            font=("Arial", 12, "bold"),
            width=15,
            height=2
        )
        self.start_button.pack(side=tk.LEFT, padx=5)

        help_button = tk.Button(
            control_frame,
            text="Help",
            command=self.show_help,
            font=("Arial", 12),
            width=10
        )
        help_button.pack(side=tk.RIGHT, padx=5)

        # Status Label
        self.status_label = tk.Label(
            control_frame,
            text="Status: Ready",
            font=("Arial", 10),
            fg="green"
        )
        self.status_label.pack(side=tk.LEFT, padx=20)

        # Create two columns
        left_frame = tk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, padx=5, fill=tk.BOTH, expand=True)

        right_frame = tk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, padx=5, fill=tk.BOTH, expand=True)

        # Transcript Section
        tk.Label(left_frame, text="Real-Time Transcript", font=("Arial", 12, "bold")).pack()
        
        self.transcript_widget = tk.Text(
            left_frame,
            wrap=tk.WORD,
            height=20,
            width=50,
            font=("Arial", 10)
        )
        self.transcript_widget.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        transcript_scroll = tk.Scrollbar(left_frame, command=self.transcript_widget.yview)
        transcript_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.transcript_widget.config(yscrollcommand=transcript_scroll.set)

        # AI Detection Section
        tk.Label(right_frame, text="AI Detection Results", font=("Arial", 12, "bold")).pack()
        
        self.detection_widget = tk.Text(
            right_frame,
            wrap=tk.WORD,
            height=20,
            width=50,
            font=("Arial", 10)
        )
        self.detection_widget.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        detection_scroll = tk.Scrollbar(right_frame, command=self.detection_widget.yview)
        detection_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.detection_widget.config(yscrollcommand=detection_scroll.set)

        # Disable text widgets
        self.transcript_widget.config(state=tk.DISABLED)
        self.detection_widget.config(state=tk.DISABLED)

    def show_help(self):
        """Show help dialog"""
        help_text = """
How to Use AI Speech Detector:

1. Click 'Start Recording' to begin
   - Ensure your microphone is connected
   - Speak clearly into the microphone

2. The left panel shows real-time transcript
   - Your speech will appear here as text

3. The right panel shows AI detection results
   - Green: Likely human speech
   - Red: Likely AI-generated speech

4. Click 'Stop Recording' to end session

Notes:
- First-time startup may take a few moments
- Keep background noise to a minimum
- Speak in complete sentences for best results

If you experience issues:
- Check microphone connection
- Restart the application
- Ensure no other apps are using the microphone
"""
        messagebox.showinfo("Help Guide", help_text)

    def toggle_recording(self):
        """Toggle between recording and stopped states"""
        if not self.is_recording:
            try:
                self.stream = self.p.open(
                    format=pyaudio.paFloat32,
                    channels=self.CHANNELS,
                    rate=self.RATE,
                    input=True,
                    frames_per_buffer=self.CHUNK
                )
                self.start_recording()
                self.start_button.config(text="Stop Recording", bg="red")
                self.status_label.config(text="Status: Recording", fg="red")
            except Exception as e:
                messagebox.showerror("Error", f"Could not start recording: {str(e)}")
                return
        else:
            self.stop_recording()
            self.start_button.config(text="Start Recording", bg="green")
            self.status_label.config(text="Status: Ready", fg="green")

    def start_recording(self):
        """Start recording and transcription"""
        self.is_recording = True
        self.stop_event.clear()
        
        # Start recording thread
        threading.Thread(target=self.audio_capture_thread).start()
        
        # Start transcription thread
        threading.Thread(target=self.transcribe_audio).start()

    def stop_recording(self):
        """Stop recording and transcription"""
        self.is_recording = False
        self.stop_event.set()
        if hasattr(self, 'stream'):
            self.stream.stop_stream()
            self.stream.close()

    def load_ai_detectors(self):
        """Load multiple AI detection models"""
        detectors = []
        try:
            # RoBERTa OpenAI Detector
            roberta_detector = pipeline('text-classification', model='roberta-base-openai-detector')
            detectors.append(roberta_detector)
        except Exception as e:
            print(f"Detector loading error: {e}")
        return detectors

    def advanced_ai_detection(self, text):
        """
        Comprehensive AI detection with multiple strategies
        """
        # Preprocess text
        text = text.strip()
        words = text.split()
        
        # Skip extremely short texts
        if len(words) < 5:
            return {
                "classification": "Insufficient Data", 
                "probability": 0.0,
                "confidence_level": "None"
            }

        # Scoring mechanisms
        detection_scores = []

        # 1. Roberta AI Detection
        if self.ai_detectors:
            try:
                roberta_result = self.ai_detectors[0](text)[0]
                detection_scores.append(roberta_result['score'])
            except Exception as e:
                print(f"RoBERTa detection error: {e}")

        # 2. Linguistic Pattern Analysis
        linguistic_score = self.analyze_linguistic_patterns(text)
        detection_scores.append(linguistic_score)

        # 3. Semantic Coherence
        semantic_score = self.check_semantic_coherence(text)
        detection_scores.append(semantic_score)

        # 4. Stylistic Analysis
        stylistic_score = self.analyze_stylistic_patterns(text)
        detection_scores.append(stylistic_score)

        # Calculate final probability
        if detection_scores:
            final_score = statistics.mean(detection_scores)
            
            # Detailed classification
            if final_score > 0.7:
                classification = "AI"
                confidence_level = "High"
            elif final_score > 0.5:
                classification = "Likely AI"
                confidence_level = "Medium"
            elif final_score > 0.3:
                classification = "Possibly AI"
                confidence_level = "Low"
            else:
                classification = "Human"
                confidence_level = "High"

            return {
                "classification": classification,
                "probability": final_score,
                "confidence_level": confidence_level,
                "detection_details": {
                    "linguistic_score": linguistic_score,
                    "semantic_score": semantic_score,
                    "stylistic_score": stylistic_score
                }
            }
        
        return {
            "classification": "Uncertain", 
            "probability": 0.0,
            "confidence_level": "None"
        }

    def analyze_linguistic_patterns(self, text):
        """Detect AI-like linguistic patterns"""
        text_lower = text.lower()
        
        # Check for AI markers
        ai_phrase_count = sum(
            1 for category in self.ai_markers.values() 
            for phrase in category 
            if phrase in text_lower
        )
        
        # Penalize repetitive language and generic statements
        unique_words = len(set(text.split()))
        total_words = len(text.split())
        
        # Calculate linguistic anomaly score
        linguistic_score = min(ai_phrase_count * 0.3, 1.0)
        linguistic_score += (1 - (unique_words / total_words)) * 0.4
        
        return min(linguistic_score, 1.0)

    def check_semantic_coherence(self, text):
        """Analyze semantic coherence and language complexity"""
        blob = TextBlob(text)
        
        # Analyze sentence structure complexity
        sentences = blob.sentences
        avg_sentence_length = np.mean([len(str(sentence).split()) for sentence in sentences])
        
        # Sentiment and subjectivity analysis
        sentiment_variation = abs(blob.sentiment.polarity)
        subjectivity = blob.sentiment.subjectivity
        
        # Combine metrics
        coherence_score = 1.0 - min(
            (sentiment_variation + subjectivity + abs(avg_sentence_length - 15) / 30), 
            1.0
        )
        
        return coherence_score

    def analyze_stylistic_patterns(self, text):
        """Analyze stylistic characteristics typical of AI-generated text"""
        # Check for overly formal or generic language
        formality_markers = [
            'comprehensive', 'innovative', 'cutting-edge', 
            'state-of-the-art', 'synergistic'
        ]
        
        formality_count = sum(
            1 for marker in formality_markers 
            if marker.lower() in text.lower()
        )
        
        # Check for perfect grammatical structure
        blob = TextBlob(text)
        grammatical_score = 1.0 - min(formality_count * 0.2, 1.0)
        
        return grammatical_score

    def setup_ui(self):
        self.root = tk.Tk()
        self.root.title("Real-Time AI Transcription Detector")
        self.root.geometry("1200x800")  # Wider window

        # Main frame with two columns
        main_frame = tk.Frame(self.root)
        main_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        # Left column - Transcript
        left_frame = tk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, padx=5, fill=tk.BOTH, expand=True)

        transcript_label = tk.Label(left_frame, text="Real-Time Transcript", font=("Arial", 12, "bold"))
        transcript_label.pack()

        self.transcript_widget = tk.Text(
            left_frame, 
            wrap=tk.WORD, 
            height=20, 
            width=50,
            font=("Arial", 10)
        )
        self.transcript_widget.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        transcript_scrollbar = tk.Scrollbar(left_frame, command=self.transcript_widget.yview)
        transcript_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.transcript_widget.config(yscrollcommand=transcript_scrollbar.set)

        # Right column - AI Detection
        right_frame = tk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, padx=5, fill=tk.BOTH, expand=True)

        detection_label = tk.Label(right_frame, text="AI Detection Results", font=("Arial", 12, "bold"))
        detection_label.pack()

        self.detection_widget = tk.Text(
            right_frame, 
            wrap=tk.WORD, 
            height=20, 
            width=50,
            font=("Arial", 10)
        )
        self.detection_widget.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        detection_scrollbar = tk.Scrollbar(right_frame, command=self.detection_widget.yview)
        detection_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.detection_widget.config(yscrollcommand=detection_scrollbar.set)

        # Configure state
        self.transcript_widget.config(state=tk.DISABLED)
        self.detection_widget.config(state=tk.DISABLED)

    def update_transcript(self, text):
        """Update real-time transcript"""
        self.root.after(0, self._update_transcript_thread, text)

    def _update_transcript_thread(self, text):
        """Thread-safe transcript update"""
        self.transcript_widget.config(state=tk.NORMAL)
        self.transcript_widget.insert(tk.END, text + " ")
        self.transcript_widget.see(tk.END)
        self.transcript_widget.config(state=tk.DISABLED)

    def update_ai_detection(self, text, result):
        """Update AI detection results"""
        self.root.after(0, self._update_ai_detection_thread, text, result)

    def _update_ai_detection_thread(self, text, result):
        """Thread-safe AI detection update"""
        self.detection_widget.config(state=tk.NORMAL)
        
        # Determine color based on classification
        color = "red" if result['classification'] in ["AI", "Likely AI"] else "green"
        
        # Format detection result
        detection_text = (
            f"Text: {text}\n"
            f"Classification: {result['classification']}\n"
            f"Probability: {result['probability']:.2f}\n"
            f"Confidence: {result['confidence_level']}\n\n"
        )
        
        # Insert with color tag
        self.detection_widget.tag_config("red", foreground="red")
        self.detection_widget.tag_config("green", foreground="green")
        
        self.detection_widget.insert(tk.END, detection_text, color)
        self.detection_widget.see(tk.END)
        self.detection_widget.config(state=tk.DISABLED)

    def process_accumulated_text(self, accumulated_text):
        """Process accumulated text for AI detection"""
        # Clean and prepare text
        clean_text = ' '.join(accumulated_text).strip()
        
        # Perform AI detection
        if clean_text:
            ai_result = self.advanced_ai_detection(clean_text)
            
            # Update UI with detection results
            self.update_ai_detection(clean_text, ai_result)
        
        # Reset buffer
        self.text_buffer.clear()
        self.last_chunk_time = time.time()

    def audio_capture_thread(self):
        """Capture audio chunks"""
        while not self.stop_event.is_set():
            try:
                audio_data = self.stream.read(self.CHUNK, exception_on_overflow=False)
                audio_np = np.frombuffer(audio_data, dtype=np.float32)
                self.audio_queue.put(audio_np)
            except Exception as e:
                print(f"Audio capture error: {e}")
                break

    def transcribe_audio(self):
        """Transcribe audio with chunk accumulation"""
        audio_buffer = []

        while not self.stop_event.is_set():
            try:
                chunk = self.audio_queue.get(timeout=1)
                audio_buffer.extend(chunk)

                if len(audio_buffer) >= self.RATE * 3:
                    audio_np = np.array(audio_buffer)
                    segments, info = self.model.transcribe(
                        audio_np, 
                        language="en", 
                        vad_filter=True,
                        beam_size=1
                    )

                    for segment in segments:
                        clean_text = segment.text.strip()
                        
                        # Update real-time transcript
                        if clean_text:
                            self.update_transcript(clean_text)
                            self.text_buffer.append(clean_text)

                            # Process text when buffer reaches threshold or timeout occurs
                            current_time = time.time()
                            word_count = len(clean_text.split())
                            
                            if (len(self.text_buffer) >= 3 or 
                                word_count >= self.MIN_CHUNK_WORDS or 
                                (self.last_chunk_time and 
                                 current_time - self.last_chunk_time > self.CHUNK_TIMEOUT)):
                                self.process_accumulated_text(self.text_buffer)

                    audio_buffer = []
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Transcription error: {e}")
                break

    def start_transcription(self):
        print("Starting real-time transcription...")
        self.stop_event.clear()

        capture_thread = threading.Thread(target=self.audio_capture_thread)
        capture_thread.start()

        transcribe_thread = threading.Thread(target=self.transcribe_audio)
        transcribe_thread.start()

        self.root.mainloop()

    def stop_transcription(self):
        print("Stopping transcription...")
        self.stop_event.set()
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

    def run(self):
        """Start the application"""
        try:
            if hasattr(self, 'root'):
                self.root.mainloop()
        except Exception as e:
            self.show_error_and_exit(f"Error running application: {str(e)}")


if __name__ == "__main__":
    try:
        app = AdvancedRealtimeTranscriber()
        app.run()
    except Exception as e:
        if not hasattr(app, 'show_error_and_exit'):
            messagebox.showerror("Fatal Error", f"Application failed to start: {str(e)}")
        sys.exit(1)