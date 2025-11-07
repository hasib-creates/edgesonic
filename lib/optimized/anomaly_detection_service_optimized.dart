import 'dart:io';
import 'dart:typed_data';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'audio_processing_service_optimized.dart';

class AnomalyResult {
  final double rawScore;
  final double smoothedScore;
  final bool isAnomaly;
  final double threshold;
  final DateTime timestamp;
  final double? timeSeconds;
  final double processingTimeMs;

  AnomalyResult({
    required this.rawScore,
    required this.smoothedScore,
    required this.isAnomaly,
    required this.threshold,
    required this.timestamp,
    this.timeSeconds,
    required this.processingTimeMs,
  });

  @override
  String toString() {
    return 'AnomalyResult(score: ${smoothedScore.toStringAsFixed(4)}, '
           'anomaly: $isAnomaly, time: ${timeSeconds?.toStringAsFixed(2)}s)';
  }
}

/// Optimized real-time anomaly detection service
/// Matches Python implementation exactly for validation
class AnomalyDetectionServiceOptimized {
  late Interpreter _interpreter;
  late AudioProcessingServiceOptimized _audioProcessor;
  
  bool _isLoaded = false;
  double threshold = 0.01; // Default, should match Python
  
  // Smoothing state (matches Python exactly)
  final List<double> _scoreHistory = [];
  static const int maxHistoryLength = 5;
  static const double smoothingAlpha = 0.6;
  
  // Performance tracking
  int _totalInferences = 0;
  double _totalInferenceTimeMs = 0.0;
  final Stopwatch _stopwatch = Stopwatch();
  
  String? lastError;

  AnomalyDetectionServiceOptimized() {
    _audioProcessor = AudioProcessingServiceOptimized();
  }

  Future<bool> loadModel({String modelPath = 'assets/models/tcn_model_int8.tflite'}) async {
    try {
      final options = InterpreterOptions()..threads = 2;
      _interpreter = await Interpreter.fromFile(
        File(modelPath),
        options: options,
      );
      
      // Verify model input/output shapes
      final inputShape = _interpreter.getInputTensor(0).shape;
      final outputShape = _interpreter.getOutputTensor(0).shape; // reconstruction
      
      print('Model loaded successfully:');
      print('  Input shape: $inputShape');
      print('  Output shape: $outputShape');
      print('  Threshold: ${threshold.toStringAsFixed(4)}');
      
      // Warm-up inference
      await _runWarmup();
      
      _isLoaded = true;
      return true;
    } catch (e) {
      lastError = 'Failed to load model: $e';
      print(lastError);
      return false;
    }
  }

  /// Run warm-up inference to initialize TFLite
  Future<void> _runWarmup() async {
    try {
      print('Running warm-up inference...');
      final dummyInput = List.generate(
        1,
        (_) => List.generate(
          AudioConfig.numMelBins,
          (_) => List.generate(AudioConfig.targetLength, (_) => 0.0),
        ),
      );

      final dummyOutput = List.generate(
        1,
        (_) => List.generate(
          AudioConfig.numMelBins,
          (_) => List.generate(AudioConfig.targetLength, (_) => 0.0),
        ),
      );

      _stopwatch.start();
      _interpreter.run(dummyInput, dummyOutput);
      _stopwatch.stop();

      print('  Warm-up inference time: ${_stopwatch.elapsedMilliseconds}ms');
      _stopwatch.reset();
    } catch (e) {
      print('  Warm-up failed: $e');
      rethrow;
    }
  }

  /// Run inference and calculate anomaly score
  /// Matches Python implementation exactly
  Future<AnomalyResult> detectAnomaly(List<List<List<double>>> inputTensor) async {
    if (!_isLoaded) {
      throw StateError('Model not loaded. Call loadModel() first.');
    }

    _stopwatch.start();

    // Prepare output buffer (reconstruction)
    // Note: tflite_flutter handles quantization/dequantization automatically
    final outputTensor = List.generate(
      1,
      (_) => List.generate(
        AudioConfig.numMelBins,
        (_) => List.generate(AudioConfig.targetLength, (_) => 0.0),
      ),
    );

    // Run inference (tflite_flutter handles INT8 quantization automatically)
    _interpreter.run(inputTensor, outputTensor);

    // Check if we need to transpose output to match input format
    // Python implementation transposes from (1, 16, 128) to (1, 128, 16)
    // But in Dart, our input is [1, 16, 128] and output should match
    // tflite_flutter typically returns in the same format as model output

    // Calculate MSE loss (matches Python: torch.mean((input - recon) ** 2))
    // Input format: [batch=1, mels=16, frames=128]
    // Output format should match for MSE calculation
    double rawScore = _calculateMSE(inputTensor[0], outputTensor[0]);

    // Apply smoothing (matches Python exactly)
    double smoothedScore;
    if (_scoreHistory.isNotEmpty) {
      smoothedScore = smoothingAlpha * rawScore +
                      (1 - smoothingAlpha) * _scoreHistory.last;
    } else {
      smoothedScore = rawScore;
    }

    // Update history
    _scoreHistory.add(rawScore);
    if (_scoreHistory.length > maxHistoryLength) {
      _scoreHistory.removeAt(0);
    }

    // Determine anomaly
    bool isAnomaly = smoothedScore > threshold;

    _stopwatch.stop();
    final processingTimeMs = _stopwatch.elapsedMicroseconds / 1000.0;
    _totalInferences++;
    _totalInferenceTimeMs += processingTimeMs;
    _stopwatch.reset();

    return AnomalyResult(
      rawScore: rawScore,
      smoothedScore: smoothedScore,
      isAnomaly: isAnomaly,
      threshold: threshold,
      timestamp: DateTime.now(),
      processingTimeMs: processingTimeMs,
    );
  }

  /// Calculate Mean Squared Error between input and reconstruction
  double _calculateMSE(List<List<double>> input, List<List<double>> output) {
    double sumSquaredError = 0.0;
    int totalElements = 0;
    
    for (int i = 0; i < input.length; i++) {
      for (int j = 0; j < input[i].length; j++) {
        final diff = input[i][j] - output[i][j];
        sumSquaredError += diff * diff;
        totalElements++;
      }
    }
    
    return sumSquaredError / totalElements;
  }

  /// Process audio chunk end-to-end (audio → mel → inference)
  Future<AnomalyResult> processAudioChunk(Float32List audioChunk) async {
    // Convert to mel spectrogram
    final melSpec = _audioProcessor.audioToMelSpectrogram(audioChunk);
    
    // Prepare model input
    final modelInput = _audioProcessor.prepareModelInput(melSpec);
    
    // Run inference
    return await detectAnomaly(modelInput);
  }

  /// Process streaming audio samples
  Stream<AnomalyResult> processStreamingAudio(Stream<Float32List> audioStream) async* {
    await for (final samples in audioStream) {
      final chunks = _audioProcessor.addAudioSamples(samples);
      
      for (final chunk in chunks) {
        yield await processAudioChunk(chunk);
      }
    }
  }

  /// Get comprehensive performance metrics
  Map<String, dynamic> getPerformanceMetrics() {
    final audioMetrics = _audioProcessor.getPerformanceMetrics();
    final avgInferenceMs = _totalInferences > 0 
        ? _totalInferenceTimeMs / _totalInferences 
        : 0.0;
    
    return {
      'total_inferences': _totalInferences,
      'avg_inference_ms': avgInferenceMs,
      'total_inference_ms': _totalInferenceTimeMs,
      'inference_fps': avgInferenceMs > 0 ? 1000.0 / avgInferenceMs : 0.0,
      'audio_processing': audioMetrics,
      'threshold': threshold,
      'score_history_length': _scoreHistory.length,
    };
  }

  /// Reset all metrics and state
  void resetMetrics() {
    _totalInferences = 0;
    _totalInferenceTimeMs = 0.0;
    _audioProcessor.resetMetrics();
    _scoreHistory.clear();
  }

  /// Clear streaming buffer
  void clearBuffer() {
    _audioProcessor.clearBuffer();
  }

  void dispose() {
    _interpreter.close();
    _scoreHistory.clear();
    _audioProcessor.clearBuffer();
  }
}
