# Validation Workflow

This folder hosts utilities for reproducing Python ↔ Android validation.

1. Place or symlink the WAV clips you want to validate inside `validation/` or
   refer to them from elsewhere in the repo.
2. Create a manifest JSON file that lists the clips you want to process.
   An example structure is provided in `manifest.example.json`.
3. Run the generator:

   ```bash
   source ../.venv/bin/activate
   python validation/generate_golden.py \
     --manifest validation/manifest.example.json \
     --model tflite/tcn_model_int8.tflite \
     --threshold 0.3 \
     --output-dir validation/golden
   ```

   This produces per-window CSVs and an aggregate `summary.json` describing the
   run. These CSVs become the golden references you can ship to Android.
4. After running Android-side validation, use `validation/compare_runs.py`
   (to be added) to check for drift by diffing CSV pairs.
