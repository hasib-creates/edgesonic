# edge_sonic_app

A new Flutter project.

## Getting Started

This project is a starting point for a Flutter application.

A few resources to get you started if this is your first Flutter project:

- [Lab: Write your first Flutter app](https://docs.flutter.dev/get-started/codelab)
- [Cookbook: Useful Flutter samples](https://docs.flutter.dev/cookbook)

For help getting started with Flutter development, view the
[online documentation](https://docs.flutter.dev/), which offers tutorials,
samples, guidance on mobile development, and a full API reference.

## Python ↔ Android Validation Checklist

1. **Prepare goldens**  
   Run `inference_realtime_tflite.py` against each WAV in `validation/`, saving a CSV that contains `window_index,time_seconds,raw_score,smoothed_score`.

2. **Process on Android**  
   Load the same `.tflite` in the app, process the matching WAV, and note the exported CSV path shown in the UI. The file lives under the app documents directory in `validation_exports/`.

3. **Pull for comparison**  
   Use `adb pull "<path from UI>" ./android_results/` to copy the CSV back to your workstation.

4. **Diff against Python**  
   Compare the Android CSV with the Python golden (max absolute diff ≤1e-3). Summaries (min/mean/max) should match the Python report.

Re-run these steps after every model conversion or major preprocessing change to confirm the Android inference path remains aligned with the Python baseline.
