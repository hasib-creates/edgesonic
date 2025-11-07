import os, matplotlib
if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
    matplotlib.use("TkAgg")
else:
    matplotlib.use("Agg")
import argparse
import json
import time
from pathlib import Path
from collections import deque
import threading
import queue
import torch
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torchaudio
import pyaudio
project_root = Path(__file__).resolve().parent

TCN_CONFIG = {
        'input_dim': 16,
        'latent_dim': 8,
        'window_size': 128,
        'chunk_size': 64
    }

AUDIO_CONFIG = {
        'sample_rate': 16000,
        'n_fft': 512,
        'hop_length': 256,
        'num_mel_bins': TCN_CONFIG['input_dim'],
        'target_length': TCN_CONFIG['window_size'],
        'norm_mean': -5.0,
        'norm_std': 4.5
    }

class RealtimeTFLiteDetector:
    """Real-time anomaly detection with a quantized INT8 TFLite model."""

    def __init__(self, model_path: Path, threshold: float = 0.01):
        print("Initializing INT8 TFLite Anomaly Detector...")
        if not model_path.exists():
            raise FileNotFoundError(f"TFLite model not found at {model_path}")

        # Load TFLite model
        self.interpreter = tf.lite.Interpreter(model_path=str(model_path))
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()
        self.input_scale, self.input_zero_point = self.input_details['quantization']

        # Heuristic: pick recon output by closest shape to input
        self.recon_output_details = None
        input_size = np.prod(self.input_details['shape'])
        min_shape_diff = float('inf')
        for details in self.output_details:
            output_size = np.prod(details['shape'])
            shape_diff = abs(input_size - output_size)
            if shape_diff < min_shape_diff:
                min_shape_diff = shape_diff
                self.recon_output_details = details
        if self.recon_output_details is None:
            raise RuntimeError("Could not find a suitable reconstruction output tensor.")
        self.output_scale, self.output_zero_point = self.recon_output_details['quantization']

        print(f"✅ Model loaded from: {model_path}")
        print(f"  Input Details: Shape={self.input_details['shape']}, Type={self.input_details['dtype']}")
        print(f"  Recon Output Details: Index={self.recon_output_details['index']}, Shape={self.recon_output_details['shape']}")

        # Configs
        self.model_config = TCN_CONFIG
        self.audio_config = AUDIO_CONFIG
        self.threshold = threshold

        # BUGFIX: must read 'window_size' (NOT 'chunk_size'), default to 128
        self.chunk_size_frames = int(self.model_config.get('window_size', 128))

        # Preprocessing transform (torchaudio)
        if torchaudio:
            self.mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.audio_config.get("sample_rate", 16000),
                n_fft=self.audio_config.get("n_fft", 512),
                hop_length=self.audio_config.get("hop_length", 256),
                n_mels=self.audio_config.get("num_mel_bins", 16),
            )
        else:
            raise RuntimeError("torchaudio is required for audio preprocessing.")

        # Derive samples_per_chunk & hop_samples (50% overlap)
        hop_length = int(self.audio_config.get("hop_length", 256))
        n_fft = int(self.audio_config.get("n_fft", 512))
        self.samples_per_chunk = (self.chunk_size_frames - 1) * hop_length + n_fft
        self.hop_samples = int(self.samples_per_chunk * 0.5)

        # One-line runtime confirmation (Req. #7)
        sr = int(self.audio_config.get("sample_rate", 16000))
        chunk_sec = self.samples_per_chunk / sr
        step_sec = self.hop_samples / sr
        print(f"Chunk: {self.chunk_size_frames} frames (≈{chunk_sec:.3f}s), step≈{step_sec:.3f}s")

        # Smoothing state
        self.score_history = deque(maxlen=5)
        self.smoothing_alpha = 0.6
        print(f"  Threshold: {self.threshold:.4f}")
        print("Detector initialized.")



    def _preprocess_audio_chunk(self, waveform: np.ndarray) -> np.ndarray:
        """Converts raw audio into a normalized float32 Mel spectrogram."""
        waveform_tensor = torch.from_numpy(waveform.copy()).float().unsqueeze(0)
        mel_spec = self.mel_transform(waveform_tensor)
        mel_spec = torch.log(mel_spec + 1e-6)
        
        mean = self.audio_config.get("norm_mean", -5.0)
        std = self.audio_config.get("norm_std", 4.5)
        normalized_mel = (mel_spec - mean) / std
        
        # Return in N, C, L format
        return normalized_mel[..., :self.chunk_size_frames].numpy()

    def detect_anomaly(self, float32_input: np.ndarray) -> dict:
        """Runs INT8 inference and calculates the anomaly score."""
        int8_input = (float32_input / self.input_scale) + self.input_zero_point
        int8_input = int8_input.astype(self.input_details['dtype'])

        self.interpreter.set_tensor(self.input_details['index'], int8_input)
        self.interpreter.invoke()

        int8_output = self.interpreter.get_tensor(self.recon_output_details['index'])
        dequantized_output_nlc = (int8_output.astype(np.float32) - self.output_zero_point) * self.output_scale
        # The output from TFLite is (1, 16, 128), but the input was (1, 128, 16).
        # We swap the last two axes (1 and 2) to align them for the MSE calculation.
        dequantized_output_ncl = np.transpose(dequantized_output_nlc, (0, 2, 1))

        raw_score = np.mean((float32_input - dequantized_output_ncl)**2)

        self.score_history.append(raw_score)
        if len(self.score_history) > 1:
            smoothed_score = (self.smoothing_alpha * raw_score +
                              (1 - self.smoothing_alpha) * self.score_history[-2])
        else:
            smoothed_score = raw_score
        
        is_anomaly = smoothed_score > self.threshold

        return {'timestamp': time.time(), 'raw_score': raw_score, 'smoothed_score': smoothed_score,
                'is_anomaly': is_anomaly, 'threshold': self.threshold,
                'input_mel': float32_input[0], 'recon_mel': dequantized_output_ncl[0]}

    def process_file_streaming(self, audio_path: Path, visualize: bool = False):
        """Processes an audio file in a streaming fashion to simulate real-time."""
        waveform, sr = torchaudio.load(audio_path)
        target_sr = self.audio_config.get("sample_rate", 16000)
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            waveform = resampler(waveform)
            
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
            
        waveform_np = waveform.squeeze(0).numpy()

        hop_length = self.audio_config.get("hop_length", 512)
        n_fft = self.audio_config.get("n_fft", 1024)
        samples_per_chunk = (self.chunk_size_frames - 1) * hop_length + n_fft
        hop_samples = int(samples_per_chunk * 0.5)
        
        results = []
        print(f"Processing '{audio_path.name}' in chunks...")
        for i in range(0, len(waveform_np) - samples_per_chunk, hop_samples):
            audio_chunk = waveform_np[i : i + samples_per_chunk]
            float32_mel = self._preprocess_audio_chunk(audio_chunk)
            result = self.detect_anomaly(float32_mel)
            result['time_seconds'] = i / target_sr
            results.append(result)
            status = "ANOMALY" if result['is_anomaly'] else "NORMAL"
            print(f"\rTime: {result['time_seconds']:.2f}s | Score: {result['smoothed_score']:.4f} | Status: {status}", end="")

        print("\nFile processing complete.")
        
        return results


class LiveAudioProcessor:
    """Handles live audio stream from microphone using PyAudio."""
    def __init__(self, detector: RealtimeTFLiteDetector, sample_rate: int = 16000):
        if pyaudio is None or torchaudio is None:
            raise ImportError("PyAudio and Torchaudio are required for live processing.")
        
        self.detector = detector
        self.sample_rate = sample_rate
        hop_length = self.detector.audio_config.get("hop_length", 512)
        n_fft = self.detector.audio_config.get("n_fft", 1024)
        self.chunk_samples = (self.detector.chunk_size_frames - 1) * hop_length + n_fft

        self.pa = pyaudio.PyAudio()
        self.stream = None
        self.is_running = False
        self.score_queue = queue.Queue()

    def start_stream(self, device_index=None):
        """Starts the audio stream and processing thread."""
        self.stream = self.pa.open(
            format=pyaudio.paFloat32, channels=1, rate=self.sample_rate,
            input=True, input_device_index=device_index, frames_per_buffer=self.chunk_samples
        )
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._process_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        self._visualize_live()

    def _process_loop(self):
        """The loop that reads from mic and runs detection."""
        while self.is_running:
            try:
                audio_data = self.stream.read(self.chunk_samples, exception_on_overflow=False)
                audio_array = np.frombuffer(audio_data, dtype=np.float32)
                if len(audio_array) < self.chunk_samples: continue
                
                float32_mel = self.detector._preprocess_audio_chunk(audio_array)
                result = self.detector.detect_anomaly(float32_mel)
                self.score_queue.put(result)
            except Exception as e:
                print(f"Error in processing loop: {e}")
                break
        
    def _visualize_live(self):
        """Live visualization of anomaly scores."""
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 6))
        times, scores = deque(maxlen=200), deque(maxlen=200)
        line, = ax1.plot([], [], 'b-')
        ax1.axhline(y=self.detector.threshold, color='r', linestyle='--', label='Threshold')
        ax1.set_ylim(0, self.detector.threshold * 5)
        ax1.set_xlabel('Time (s)'); ax1.set_ylabel('Anomaly Score'); ax1.legend()
        ax1.set_title("Live Anomaly Detection (INT8 Model)")
        start_time = time.time()

        def update_plot(frame):
            while not self.score_queue.empty():
                try:
                    result = self.score_queue.get_nowait()
                    times.append(result['timestamp'] - start_time)
                    scores.append(result['smoothed_score'])
                    fig.patch.set_facecolor('#ffcccc' if result['is_anomaly'] else 'white')
                except queue.Empty:
                    break
            if times:
                line.set_data(times, scores)
                ax1.set_xlim(max(0, times[-1] - 20), times[-1] + 1)
                ax1.set_ylim(0, max(self.detector.threshold * 1.5, max(scores) * 1.2 if scores else 1))
            return line,
        
        ani = FuncAnimation(fig, update_plot, interval=100, blit=True, cache_frame_data=False)
        try:
            plt.show()
        except KeyboardInterrupt:
            self.stop_stream()
        finally:
            self.stop_stream()

    def stop_stream(self):
        """Stops the audio stream and closes PyAudio."""
        if self.is_running:
            self.is_running = False
            if hasattr(self, 'processing_thread') and self.processing_thread.is_alive():
                self.processing_thread.join(timeout=1)
            if self.stream: self.stream.close()
            self.pa.terminate()
            print("\nAudio stream stopped.")

def main_app():
    parser = argparse.ArgumentParser(
        description='Real-time anomaly detection with a quantized TFLite model.'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default="/home/hasib/TestTF/tcn_model_air/aircon/tcn_model_int8.tflite",
        help='Path to the .tflite model file.'
    )
    parser.add_argument('--input', type=str, required=True, help='Input audio file path or "live" for microphone.')
    parser.add_argument('--threshold', type=float, default=0.01, help='Anomaly score threshold.')
    parser.add_argument('--save_plot', type=str, default=None, help='If set, save the plot to this PNG instead of showing a window.')
    parser.add_argument('--visualize', action='store_true', help='Visualize results for file processing.')
    
    args = parser.parse_args()

    try:
        detector = RealtimeTFLiteDetector(Path(args.model_path), args.threshold)
        if args.input.lower() == 'live':
            LiveAudioProcessor(detector).start_stream()
        else:
            input_path = Path(args.input)
            if input_path.is_file():
                results = detector.process_file_streaming(input_path, args.visualize)
                if args.visualize and results:
                    times  = [r['time_seconds']   for r in results]
                    scores = [r['smoothed_score'] for r in results]

                    s_min = float(np.min(scores))
                    s_max = float(np.max(scores))
                    s_mean = float(np.mean(scores))
                    t_max  = times[int(np.argmax(scores))]

                    print(f"\nSummary — min: {s_min:.6f}, mean: {s_mean:.6f}, max: {s_max:.6f} at t={t_max:.2f}s")

                    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
                    ax.plot(times, scores, label='Anomaly Score')

                    # Threshold
                    ax.axhline(y=detector.threshold, color='r', linestyle='--', label='Threshold')

                    # Min / Mean / Max reference lines
                    ax.axhline(y=s_mean, color='tab:orange', linestyle=':', linewidth=1.5, label=f"Mean = {s_mean:.4f}")
                    ax.axhline(y=s_min,  color='tab:green',  linestyle=':', linewidth=1.0, label=f"Min = {s_min:.4f}")
                    ax.axhline(y=s_max,  color='tab:purple', linestyle=':', linewidth=1.0, label=f"Max = {s_max:.4f}")

                    # Peak marker
                    ax.plot([t_max], [s_max], marker='o', markersize=7, color='tab:purple', label='Peak')

                    ax.set_xlabel('Time (s)')
                    ax.set_ylabel('Score')
                    ax.set_title('Anomaly Detection Results')
                    ax.grid(True)

                    # Keep legend clean (deduplicate)
                    handles, labels = ax.get_legend_handles_labels()
                    uniq = dict(zip(labels, handles))
                    ax.legend(uniq.values(), uniq.keys(), loc='best')

                    # Tidy y-limits to show everything clearly
                    y_top = max(s_max, detector.threshold) * 1.15 if max(s_max, detector.threshold) > 0 else 0.1
                    y_bot = min(0.0, min(scores) * 0.9)
                    ax.set_ylim(y_bot, y_top)

                    plt.tight_layout()
                    
                    # Always save the plot to current directory with timestamp
                    save_filename = f"anomaly_plot_{int(time.time())}.png"
                    plt.savefig(save_filename, dpi=300, bbox_inches='tight')
                    print(f"Plot saved to: {save_filename}")
                    
                    # Also try to show it if display is available
                    try:
                        plt.show()
                    except Exception as e:
                        print(f"Could not display plot (no GUI available): {e}")
                        print(f"But plot is saved as: {save_filename}")

            else:
                print(f"Error: Input file not found at {input_path}")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main_app()