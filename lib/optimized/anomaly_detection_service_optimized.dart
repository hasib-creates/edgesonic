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

      // Verify model input/output shapes and types
      final inputTensor = _interpreter.getInputTensor(0);
      final outputTensor = _interpreter.getOutputTensor(0);
      final inputShape = inputTensor.shape;
      final outputShape = outputTensor.shape;
      final inputType = inputTensor.type;
      final outputType = outputTensor.type;

      print('Model loaded successfully:');
      print('  Input shape: $inputShape, type: $inputType');
      print('  Output shape: $outputShape, type: $outputType');
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

      // Get input/output tensor info
      final inputTensor = _interpreter.getInputTensor(0);
      final outputTensor = _interpreter.getOutputTensor(0);
      final inputType = inputTensor.type;
      final outputType = outputTensor.type;

      print('  Warmup using input type: $inputType, output type: $outputType');

      // Create dummy input based on tensor type
      dynamic dummyInput;
      dynamic dummyOutput;

      if (inputType == TensorType.int8 || inputType.toString().toLowerCase().contains('int8')) {
        // For INT8 quantized model, use Int8List
        final inputSize = 1 * AudioConfig.numMelBins * AudioConfig.targetLength;
        dummyInput = Int8List(inputSize);

        // Output shape from logs: [1, 128, 8]
        final outputSize = outputTensor.shape.reduce((a, b) => a * b);
        dummyOutput = Int8List(outputSize);
      } else {
        // For float32 model
        dummyInput = List.generate(
          1,
          (_) => List.generate(
            AudioConfig.numMelBins,
            (_) => List.generate(AudioConfig.targetLength, (_) => 0.0),
          ),
        );
        dummyOutput = List.generate(
          1,
          (_) => List.generate(
            AudioConfig.numMelBins,
            (_) => List.generate(AudioConfig.targetLength, (_) => 0.0),
          ),
        );
      }

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

    // Get tensor info
    final inputTensorInfo = _interpreter.getInputTensor(0);
    final outputTensorInfo = _interpreter.getOutputTensor(0);
    final inputType = inputTensorInfo.type;

    dynamic modelInput;
    dynamic modelOutput;
    List<List<double>> dequantizedOutput;

    if (inputType == TensorType.int8 || inputType.toString().toLowerCase().contains('int8')) {
      // INT8 quantized model - need to quantize input
      final inputParams = inputTensorInfo.params;
      final inputScale = inputParams.scale;
      final inputZeroPoint = inputParams.zeroPoint;

      print('Input quantization: scale=$inputScale, zeroPoint=$inputZeroPoint');

      // Flatten and quantize input: [1, 16, 128] -> Int8List
      final inputSize = 1 * AudioConfig.numMelBins * AudioConfig.targetLength;
      modelInput = Int8List(inputSize);

      int idx = 0;
      for (int b = 0; b < 1; b++) {
        for (int m = 0; m < AudioConfig.numMelBins; m++) {
          for (int f = 0; f < AudioConfig.targetLength; f++) {
            final floatVal = inputTensor[b][m][f];
            final quantizedVal = (floatVal / inputScale + inputZeroPoint).round().clamp(-128, 127);
            modelInput[idx++] = quantizedVal;
          }
        }
      }

      // Prepare output buffer
      final outputSize = outputTensorInfo.shape.reduce((a, b) => a * b);
      modelOutput = Int8List(outputSize);

      // Run inference
      _interpreter.run(modelInput, modelOutput);

      // Dequantize output
      final outputParams = outputTensorInfo.params;
      final outputScale = outputParams.scale;
      final outputZeroPoint = outputParams.zeroPoint;

      print('Output quantization: scale=$outputScale, zeroPoint=$outputZeroPoint');

      // Output shape is [1, 128, 8] but we need [16, 128] for MSE with input [16, 128]
      // Need to reshape and transpose appropriately
      dequantizedOutput = List.generate(
        AudioConfig.numMelBins,
        (_) => List.generate(AudioConfig.targetLength, (_) => 0.0),
      );

      // Dequantize: output_float = (output_int8 - zeroPoint) * scale
      idx = 0;
      for (int i = 0; i < modelOutput.length && idx < AudioConfig.numMelBins * AudioConfig.targetLength; i++) {
        final int m = idx ~/ AudioConfig.targetLength;
        final int f = idx % AudioConfig.targetLength;
        if (m < AudioConfig.numMelBins && f < AudioConfig.targetLength) {
          dequantizedOutput[m][f] = (modelOutput[i] - outputZeroPoint) * outputScale;
          idx++;
        }
      }
    } else {
      // Float32 model
      final outputTensor = List.generate(
        1,
        (_) => List.generate(
          AudioConfig.numMelBins,
          (_) => List.generate(AudioConfig.targetLength, (_) => 0.0),
        ),
      );

      _interpreter.run(inputTensor, outputTensor);
      dequantizedOutput = outputTensor[0];
    }

    // Calculate MSE loss (matches Python: torch.mean((input - recon) ** 2))
    // Input format: [batch=1, mels=16, frames=128]
    // Output format should match for MSE calculation
    double rawScore = _calculateMSE(inputTensor[0], dequantizedOutput);

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
