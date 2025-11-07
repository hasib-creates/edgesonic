#!/usr/bin/env python3
"""
Validation script to verify Flutter implementation matches Python implementation.
Tests:
1. Audio preprocessing (mel spectrogram generation)
2. TFLite inference
3. MSE calculation
4. Smoothing logic
5. End-to-end anomaly detection
"""
import argparse
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
import torchaudio
import torchaudio.transforms as T
from collections import deque

# Config (must match Flutter exactly)
AUDIO_CONFIG = {
    'sample_rate': 16000,
    'n_fft': 512,
    'hop_length': 256,
    'num_mel_bins': 16,
    'target_length': 128,
    'norm_mean': -5.0,
    'norm_std': 4.5
}

DETECTION_CONFIG = {
    'threshold': 0.01,
    'smoothing_alpha': 0.6,
    'history_length': 5
}

class PythonTFLiteDetector:
    """Python implementation matching Flutter exactly"""
    
    def __init__(self, model_path: Path, threshold: float = 0.01):
        # Load TFLite model
        self.interpreter = tf.lite.Interpreter(model_path=str(model_path))
        self.interpreter.allocate_tensors()
        
        # Get input and output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        print("TFLite Model loaded:")
        print(f"  Input shape: {self.input_details[0]['shape']}")
        print(f"  Output shape: {self.output_details[0]['shape']}")
        print(f"  Input dtype: {self.input_details[0]['dtype']}")
        print(f"  Output dtype: {self.output_details[0]['dtype']}")
        
        # Mel transform
        self.mel_transform = T.MelSpectrogram(
            sample_rate=AUDIO_CONFIG['sample_rate'],
            n_fft=AUDIO_CONFIG['n_fft'],
            hop_length=AUDIO_CONFIG['hop_length'],
            n_mels=AUDIO_CONFIG['num_mel_bins'],
        )
        
        # Detection parameters
        self.threshold = threshold
        self.score_history = deque(maxlen=DETECTION_CONFIG['history_length'])
        self.smoothing_alpha = DETECTION_CONFIG['smoothing_alpha']
        
        # Warm-up
        self._warmup()
    
    def _warmup(self):
        """Warm-up inference"""
        print("Running warm-up inference...")
        dummy_input = np.zeros(
            (1, AUDIO_CONFIG['num_mel_bins'], AUDIO_CONFIG['target_length']), 
            dtype=np.float32
        )
        self._run_inference(dummy_input)
        print("  Warm-up complete")
    
    def _preprocess_audio(self, waveform: np.ndarray) -> np.ndarray:
        """Convert raw audio to normalized mel spectrogram"""
        # Convert to tensor
        waveform_tensor = tf.convert_to_tensor(waveform, dtype=tf.float32)
        if len(waveform_tensor.shape) == 1:
            waveform_tensor = waveform_tensor[None, :]
        
        # Compute mel spectrogram
        mel_spec = self.mel_transform(waveform_tensor)
        mel_spec = np.log(mel_spec.numpy() + 1e-6)
        
        # Normalize
        mel_spec = (mel_spec - AUDIO_CONFIG['norm_mean']) / AUDIO_CONFIG['norm_std']
        
        # Ensure correct shape [1, num_mels, target_length]
        if mel_spec.shape[-1] < AUDIO_CONFIG['target_length']:
            pad_width = ((0, 0), (0, 0), (0, AUDIO_CONFIG['target_length'] - mel_spec.shape[-1]))
            mel_spec = np.pad(mel_spec, pad_width, mode='constant')
        else:
            mel_spec = mel_spec[..., :AUDIO_CONFIG['target_length']]
        
        return mel_spec.astype(np.float32)
    
    def _run_inference(self, input_tensor: np.ndarray) -> np.ndarray:
        """Run TFLite inference"""
        # Handle quantization if needed
        input_scale, input_zero_point = self.input_details[0].get('quantization', (0.0, 0))
        output_scale, output_zero_point = self.output_details[0].get('quantization', (0.0, 0))
        
        # Quantize input if int8 model
        if self.input_details[0]['dtype'] == np.int8:
            input_tensor = (input_tensor / input_scale + input_zero_point).astype(np.int8)
        
        # Set input
        self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor)
        
        # Run inference
        self.interpreter.invoke()
        
        # Get output
        output_tensor = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        # Dequantize output if int8 model
        if self.output_details[0]['dtype'] == np.int8:
            output_tensor = (output_tensor.astype(np.float32) - output_zero_point) * output_scale
        
        return output_tensor
    
    def _calculate_mse(self, input_tensor: np.ndarray, output_tensor: np.ndarray) -> float:
        """Calculate MSE between input and reconstruction"""
        mse = np.mean((input_tensor - output_tensor) ** 2)
        return float(mse)
    
    def detect_anomaly(self, waveform: np.ndarray) -> dict:
        """End-to-end anomaly detection"""
        # Preprocess
        mel_spec = self._preprocess_audio(waveform)
        
        # Inference
        reconstruction = self._run_inference(mel_spec)
        
        # Calculate raw score
        raw_score = self._calculate_mse(mel_spec, reconstruction)
        
        # Apply smoothing (matches Flutter exactly)
        if len(self.score_history) > 0:
            smoothed_score = (self.smoothing_alpha * raw_score + 
                            (1 - self.smoothing_alpha) * self.score_history[-1])
        else:
            smoothed_score = raw_score
        
        self.score_history.append(raw_score)
        
        # Determine anomaly
        is_anomaly = smoothed_score > self.threshold
        
        return {
            'raw_score': raw_score,
            'smoothed_score': smoothed_score,
            'is_anomaly': is_anomaly,
            'threshold': self.threshold,
            'mel_spec_shape': mel_spec.shape,
            'mel_spec_stats': {
                'min': float(np.min(mel_spec)),
                'max': float(np.max(mel_spec)),
                'mean': float(np.mean(mel_spec)),
                'std': float(np.std(mel_spec)),
            }
        }

def generate_test_cases(output_dir: Path):
    """Generate test audio samples and expected outputs"""
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Test case 1: Pure silence
    silence = np.zeros(32768, dtype=np.float32)
    
    # Test case 2: Pure sine wave (440 Hz)
    t = np.linspace(0, 2, 32000)
    sine = np.sin(2 * np.pi * 440 * t).astype(np.float32) * 0.5
    
    # Test case 3: White noise
    noise = np.random.randn(32768).astype(np.float32) * 0.1
    
    # Test case 4: Complex signal
    complex_signal = (
        np.sin(2 * np.pi * 440 * t) * 0.3 +
        np.sin(2 * np.pi * 880 * t) * 0.2 +
        np.random.randn(len(t)) * 0.05
    ).astype(np.float32)
    
    test_cases = [
        ('silence', silence),
        ('sine_440hz', sine),
        ('white_noise', noise),
        ('complex_signal', complex_signal)
    ]
    
    # Save test cases
    for name, waveform in test_cases:
        np.save(output_dir / f'{name}.npy', waveform)
    
    print(f"Generated {len(test_cases)} test cases in {output_dir}")
    return test_cases

def validate_implementation(model_path: Path, test_cases_dir: Path):
    """Run validation tests"""
    print("\n" + "="*60)
    print("VALIDATION TEST: Python TFLite vs Flutter Implementation")
    print("="*60 + "\n")
    
    # Load model
    detector = PythonTFLiteDetector(model_path, threshold=0.01)
    
    # Generate test cases if not exists
    if not test_cases_dir.exists():
        test_cases = generate_test_cases(test_cases_dir)
    else:
        # Load existing test cases
        test_cases = []
        for npy_file in sorted(test_cases_dir.glob('*.npy')):
            name = npy_file.stem
            waveform = np.load(npy_file)
            test_cases.append((name, waveform))
    
    # Run tests
    results = []
    for name, waveform in test_cases:
        print(f"\nTest case: {name}")
        print(f"  Waveform shape: {waveform.shape}")
        print(f"  Waveform stats: min={np.min(waveform):.4f}, max={np.max(waveform):.4f}, mean={np.mean(waveform):.4f}")
        
        result = detector.detect_anomaly(waveform)
        results.append({'name': name, **result})
        
        print(f"  Raw score: {result['raw_score']:.6f}")
        print(f"  Smoothed score: {result['smoothed_score']:.6f}")
        print(f"  Is anomaly: {result['is_anomaly']}")
        print(f"  Mel spec stats: {result['mel_spec_stats']}")
    
    # Save results for Flutter comparison
    output_file = test_cases_dir / 'expected_results.json'
    with open(output_file, 'w') as f:
        json.dump({
            'config': {
                'audio': AUDIO_CONFIG,
                'detection': DETECTION_CONFIG
            },
            'results': results
        }, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {output_file}")
    print("="*60)
    print("\nNow run the Flutter validation test with these expected results.")
    print("The Flutter implementation should produce identical scores.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Validate Flutter implementation against Python')
    parser.add_argument('--model_path', type=str, required=True, help='Path to TFLite model')
    parser.add_argument('--test_dir', type=str, default='./validation_tests', help='Directory for test cases')
    args = parser.parse_args()
    
    validate_implementation(Path(args.model_path), Path(args.test_dir))
