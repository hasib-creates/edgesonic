# 🚀 EdgeSonic Android - Quick Start

## Run in 3 Steps

### 1. Install Dependencies
```bash
cd /home/user/edgesonic
flutter pub get
```

### 2. Connect Android Device
```bash
# Enable USB debugging on your device
# Connect via USB
flutter devices  # Verify device detected
```

### 3. Run the App
```bash
# Option A: Replace main.dart temporarily
cp lib/main.dart lib/main_old.dart
cp lib/main_optimized.dart lib/main.dart
flutter run

# Option B: Modify main.dart to use optimized entry
# Edit lib/main.dart and import main_optimized.dart
```

## 🎤 Test Live Inference

1. Launch app on device
2. Grant microphone permission
3. Tap **"Start Live Capture"**
4. Speak or play audio
5. Watch real-time anomaly scores!

## 📱 Features

- ✅ **Live microphone inference** (real-time)
- ✅ **Audio file upload** (coming soon: full decoding)
- ✅ **MQTT integration** (ESP32 telemetry)
- ✅ **Optimized for Android** (5-15ms latency)

## 🔧 Troubleshooting

**Model not found?**
- Ensure `assets/models/tcn_model_int8.tflite` exists
- Run `flutter pub get` again

**Microphone permission denied?**
- Go to Settings → Apps → EdgeSonic → Permissions
- Enable Microphone

**Scores don't match Python?**
- Check `/home/user/edgesonic/ANDROID_IMPLEMENTATION_GUIDE.md`
- Verify tensor shapes with debug logging

## 📖 Full Documentation

See `ANDROID_IMPLEMENTATION_GUIDE.md` for:
- Detailed architecture
- Implementation comparison with Python
- Performance optimization guide
- Troubleshooting

## 🎯 Key Files

- `lib/main_optimized.dart` - App entry point
- `optimized/anomaly_detection_service_optimized.dart` - Inference engine
- `optimized/audio_processing_service_optimized.dart` - Audio preprocessing
- `services/live_audio_service_optimized.dart` - Microphone capture

---

**Questions?** Check the full guide or open an issue!
