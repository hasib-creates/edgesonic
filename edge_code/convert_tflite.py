# convert_tcn_esp32.py
"""
This script converts a PyTorch TCN model to an ESP32-S3 ready,
fully int8-quantized TFLite model using a 2-step process.
1. Convert PyTorch -> ONNX -> TensorFlow float32 SavedModel.
2. Quantize the float32 SavedModel -> int8 TFLite using the TensorFlow API.
"""
import sys
import subprocess
from pathlib import Path
import argparse
import shutil

import torch
import torchaudio
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from model_tcn_esp32 import TCNAutoencoderLite
from audio_dataset import SimpleAudioDataset

# These values are taken from 'train_tcn_esp32.py' script to ensure
# the conversion process matches the trained model's architecture.

TCN_CONFIG = {
    "input_dim": 16,
    "latent_dim": 8,
    "window_size": 128,
    "encoder_hidden": [8, 12],
    "decoder_hidden": [12],
    "kernel_size": 3,
}

AUDIO_CONFIG = {
    "sample_rate": 16000,
    "n_fft": 512,
    "hop_length": 256,
    "num_mel_bins": 16,      
    "target_length": 128,     
    "norm_mean": -5.0,
    "norm_std": 4.5,
}

def quantize_saved_model_to_int8(
    saved_model_dir: Path,
    representative_dataset: np.ndarray,
    output_tflite_path: Path
):
    """Quantizes a float32 SavedModel to a fully int8 TFLite model."""
    print("\nStarting Step 2: Quantizing SavedModel to INT8 TFLite...")

    def representative_dataset_gen():
        for i in range(representative_dataset.shape[0]):
            # Yield a single sample with a batch dimension of 1
            yield [np.expand_dims(representative_dataset[i], axis=0).astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    print("   Running TFLite converter and quantizing...")
    try:
        tflite_quant_model = converter.convert()
        with open(output_tflite_path, 'wb') as f:
            f.write(tflite_quant_model)

        print("\n" + "="*50)
        print("SUCCESS: INT8 quantization complete!")
        print(f"Final ESP32-S3 ready model saved at: {output_tflite_path}")
        print("="*50 + "\n")

    except Exception as e:
        print(f"\nTFLite conversion failed during quantization.")
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()


def generate_rep_dataset_manually(raw_data_dir: Path, num_samples: int):
    """Generates a representative dataset from raw audio files using the hardcoded configs."""
    print("\nGenerating representative dataset manually...")
    dataset = SimpleAudioDataset(data_dir=raw_data_dir, split="train", audio_config=AUDIO_CONFIG)
    normal_indices = dataset.get_normal_indices()

    if not normal_indices:
        raise ValueError("No 'normal' samples found in the training dataset.")

    target_frames = TCN_CONFIG["window_size"]          # 128
    hop_length = AUDIO_CONFIG["hop_length"]            # 256
    samples_per_chunk = (target_frames - 1) * hop_length + AUDIO_CONFIG["n_fft"]

    all_mel_chunks = []

    for file_idx in tqdm(normal_indices, desc="   Processing audio files"):
        if len(all_mel_chunks) >= num_samples:
            break

        waveform, _ = torchaudio.load(dataset.files[file_idx])
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # non-overlapping chunks for rep-data
        for start in range(0, waveform.shape[1] - samples_per_chunk, samples_per_chunk):
            if len(all_mel_chunks) >= num_samples:
                break

            audio_chunk = waveform[:, start : start + samples_per_chunk]
            mel_spec = dataset.mel_transform(audio_chunk)
            mel_spec = torch.log(mel_spec + 1e-6)

            mean = AUDIO_CONFIG["norm_mean"]
            std = AUDIO_CONFIG["norm_std"]
            mel_spec = (mel_spec - mean) / std

            # Require full window
            if mel_spec.shape[-1] < target_frames:
                continue

            mel_chunk = mel_spec[..., :target_frames]   # (1, 16, 128)
            all_mel_chunks.append(mel_chunk.numpy().squeeze(0))  # (16, 128)

    rep_data = np.array(all_mel_chunks, dtype=np.float32)       # (N, 16, 128)
    print(f"Created representative dataset with shape: {rep_data.shape}")
    return rep_data



def convert_model_for_mcu(model_path: Path, raw_data_dir: Path, output_dir: Path, num_samples: int):
    """Main function to run the full conversion pipeline."""
    output_dir.mkdir(exist_ok=True, parents=True)

    # --- 1. Load PyTorch Model ---
    device = torch.device("cpu")
    checkpoint = torch.load(model_path, map_location=device)

    model = TCNAutoencoderLite(
        input_dim=TCN_CONFIG["input_dim"],
        latent_dim=TCN_CONFIG["latent_dim"],
        encoder_hidden=TCN_CONFIG["encoder_hidden"],
        decoder_hidden=TCN_CONFIG["decoder_hidden"],
        kernel_size=TCN_CONFIG["kernel_size"]
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    print(f"   PyTorch model loaded from: {model_path}")
    print(f"   Model configured with Input Dim: {TCN_CONFIG['input_dim']}, Window Size: {TCN_CONFIG['window_size']}")

    # --- 2. Export to ONNX ---
    dummy_input = torch.randn(1, TCN_CONFIG["input_dim"], TCN_CONFIG["window_size"])  # (1, 16, 128)
    onnx_path = output_dir / "tcn_for_conversion.onnx"
    input_name = "input"
    torch.onnx.export(
        model, dummy_input, str(onnx_path),
        input_names=[input_name], output_names=["latent", "recon"], opset_version=11,
        dynamic_axes={'input': {0: 'batch_size'}, 'latent': {0: 'batch_size'}, 'recon': {0: 'batch_size'}}
    )
    print(f" Model exported to ONNX: {onnx_path}")

    # --- 3. ONNX -> TF SavedModel ---
    print("\n Starting Step 1: ONNX to float32 TensorFlow SavedModel conversion...")
    saved_model_path = output_dir / "tcn_saved_model"
    if saved_model_path.exists():
        shutil.rmtree(saved_model_path)
    try:
        # Explicit input shape 1,16,128
        input_shape = f"1,{TCN_CONFIG['input_dim']},{TCN_CONFIG['window_size']}"
        command = [
            "onnx2tf",
            "-i", str(onnx_path),
            "-o", str(saved_model_path),
            "-osd",
            "-ois", f"{input_name}:{input_shape}",
            "-k", input_name
        ]
        print(f"   Running onnx2tf with command: {' '.join(command)}")
        subprocess.run(command, check=True, capture_output=True, text=True)
        print(f" Successfully created float32 SavedModel at: {saved_model_path}")
    except subprocess.CalledProcessError as e:
        print(f" Failed to convert ONNX to SavedModel. Error:\n{e.stderr}")
        return

    # --- 4. Representative Dataset (N,16,128) ---
    rep_data = generate_rep_dataset_manually(raw_data_dir, num_samples)
    expected_shape = (num_samples, TCN_CONFIG['input_dim'], TCN_CONFIG['window_size'])
    if rep_data.shape != expected_shape:
        print(f"  CRITICAL ERROR: Representative dataset shape {rep_data.shape} does not match expected {expected_shape}.")
        return

    # --- 5. Quantize to TFLite (INT8) ---
    final_model_path = output_dir / "tcn_model_int8.tflite"
    quantize_saved_model_to_int8(saved_model_path, rep_data, final_model_path)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert a PyTorch model to an ESP32-S3 ready INT8 TFLite model.")
    parser.add_argument('--model_path', type=str, default="esp32_models/fan/best_model.pth", help="Path to the trained PyTorch model (.pth file).")
    parser.add_argument('--raw_data_dir', type=str, default="./data", help="Path to the root directory of your raw audio dataset (e.g., './data').")
    parser.add_argument('--output_dir', type=str, default="converted_models_final", help="Directory to save the final ONNX and TFLite files.")
    parser.add_argument('--num_samples', type=int, default=100, help="Number of samples for the representative dataset.")
    args = parser.parse_args()

    convert_model_for_mcu(
        model_path=Path(args.model_path),
        raw_data_dir=Path(args.raw_data_dir),
        output_dir=Path(args.output_dir),
        num_samples=args.num_samples
    )