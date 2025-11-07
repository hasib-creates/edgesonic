import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';
import 'package:flutter_test/flutter_test.dart';
import 'package:edge_sonic_app/services/anomaly_detection_service_optimized.dart';
import 'package:edge_sonic_app/services/audio_processing_service_optimized.dart';

/// Validation test to ensure Flutter implementation matches Python exactly
/// 
/// Usage:
/// 1. Run Python: python validate_flutter_implementation.py --model_path your_model.tflite
/// 2. Copy expected_results.json and test .npy files to test/validation_tests/
/// 3. Run: flutter test test/validation_test.dart
void main() {
  group('Flutter vs Python Implementation Validation', () {
    late AnomalyDetectionServiceOptimized detector;
    late Map<String, dynamic> expectedResults;
    
    setUpAll(() async {
      // Load expected results from Python
      final resultsFile = File('test/validation_tests/expected_results.json');
      if (!resultsFile.existsSync()) {
        fail('Expected results file not found. Run Python validation script first.');
      }
      
      final resultsJson = await resultsFile.readAsString();
      expectedResults = jsonDecode(resultsJson);
      
      // Initialize detector
      detector = AnomalyDetectionServiceOptimized();
      final loaded = await detector.loadModel(
        modelPath: 'assets/models/tcn_model_int8.tflite'
      );
      
      expect(loaded, isTrue, reason: 'Model should load successfully');
      
      // Set threshold from Python config
      detector.threshold = expectedResults['config']['detection']['threshold'];
    });
    
    tearDownAll(() {
      detector.dispose();
    });
    
    test('Audio processing configuration matches Python', () {
      expect(AudioConfig.sampleRate, equals(16000));
      expect(AudioConfig.nFft, equals(512));
      expect(AudioConfig.hopLength, equals(256));
      expect(AudioConfig.numMelBins, equals(16));
      expect(AudioConfig.targetLength, equals(128));
      expect(AudioConfig.normMean, equals(-5.0));
      expect(AudioConfig.normStd, equals(4.5));
    });
    
    test('Mel spectrogram generation matches Python', () async {
      final testCase = expectedResults['results'][0]; // silence test
      final waveform = await _loadNumpyArray('test/validation_tests/silence.npy');
      
      final audioProcessor = AudioProcessingServiceOptimized();
      final melSpec = audioProcessor.audioToMelSpectrogram(waveform);
      
      // Check shape
      expect(melSpec.length, equals(16), reason: 'Should have 16 mel bins');
      expect(melSpec[0].length, greaterThanOrEqualTo(128), reason: 'Should have at least 128 frames');
      
      // Check statistics match (within tolerance)
      final melFlat = melSpec.expand((row) => row).toList();
      final melMin = melFlat.reduce((a, b) => a < b ? a : b);
      final melMax = melFlat.reduce((a, b) => a > b ? a : b);
      final melMean = melFlat.reduce((a, b) => a + b) / melFlat.length;
      
      final pythonStats = testCase['mel_spec_stats'];
      expect(melMin, closeTo(pythonStats['min'], 0.01), reason: 'Mel min should match');
      expect(melMax, closeTo(pythonStats['max'], 0.01), reason: 'Mel max should match');
      expect(melMean, closeTo(pythonStats['mean'], 0.01), reason: 'Mel mean should match');
    });
    
    test('Anomaly detection scores match Python - Silence', () async {
      detector.resetMetrics();
      
      final testCase = expectedResults['results']
          .firstWhere((r) => r['name'] == 'silence');
      final waveform = await _loadNumpyArray('test/validation_tests/silence.npy');
      
      final result = await detector.processAudioChunk(waveform);
      
      print('Flutter result: raw=${result.rawScore}, smoothed=${result.smoothedScore}');
      print('Python result: raw=${testCase['raw_score']}, smoothed=${testCase['smoothed_score']}');
      
      // Scores should match within tolerance (0.1% relative error)
      final rawTolerance = testCase['raw_score'] * 0.001;
      final smoothedTolerance = testCase['smoothed_score'] * 0.001;
      
      expect(result.rawScore, closeTo(testCase['raw_score'], rawTolerance),
          reason: 'Raw score should match Python');
      expect(result.smoothedScore, closeTo(testCase['smoothed_score'], smoothedTolerance),
          reason: 'Smoothed score should match Python');
      expect(result.isAnomaly, equals(testCase['is_anomaly']),
          reason: 'Anomaly detection should match Python');
    });
    
    test('Anomaly detection scores match Python - Sine wave', () async {
      detector.resetMetrics();
      
      final testCase = expectedResults['results']
          .firstWhere((r) => r['name'] == 'sine_440hz');
      final waveform = await _loadNumpyArray('test/validation_tests/sine_440hz.npy');
      
      final result = await detector.processAudioChunk(waveform);
      
      final rawTolerance = testCase['raw_score'] * 0.001;
      final smoothedTolerance = testCase['smoothed_score'] * 0.001;
      
      expect(result.rawScore, closeTo(testCase['raw_score'], rawTolerance));
      expect(result.smoothedScore, closeTo(testCase['smoothed_score'], smoothedTolerance));
      expect(result.isAnomaly, equals(testCase['is_anomaly']));
    });
    
    test('Anomaly detection scores match Python - White noise', () async {
      detector.resetMetrics();
      
      final testCase = expectedResults['results']
          .firstWhere((r) => r['name'] == 'white_noise');
      final waveform = await _loadNumpyArray('test/validation_tests/white_noise.npy');
      
      final result = await detector.processAudioChunk(waveform);
      
      final rawTolerance = testCase['raw_score'] * 0.001;
      final smoothedTolerance = testCase['smoothed_score'] * 0.001;
      
      expect(result.rawScore, closeTo(testCase['raw_score'], rawTolerance));
      expect(result.smoothedScore, closeTo(testCase['smoothed_score'], smoothedTolerance));
      expect(result.isAnomaly, equals(testCase['is_anomaly']));
    });
    
    test('Smoothing algorithm matches Python across multiple samples', () async {
      detector.resetMetrics();
      
      // Process all test cases in sequence to test smoothing
      final testCases = expectedResults['results'];
      
      for (final testCase in testCases) {
        final waveform = await _loadNumpyArray(
            'test/validation_tests/${testCase['name']}.npy');
        final result = await detector.processAudioChunk(waveform);
        
        // Smoothed scores should match
        final tolerance = testCase['smoothed_score'] * 0.001;
        expect(result.smoothedScore, closeTo(testCase['smoothed_score'], tolerance),
            reason: 'Smoothed score for ${testCase['name']} should match Python');
      }
    });
    
    test('Performance metrics are tracked correctly', () async {
      detector.resetMetrics();
      
      final waveform = await _loadNumpyArray('test/validation_tests/silence.npy');
      
      // Process 10 chunks
      for (int i = 0; i < 10; i++) {
        await detector.processAudioChunk(waveform);
      }
      
      final metrics = detector.getPerformanceMetrics();
      
      expect(metrics['total_inferences'], equals(10));
      expect(metrics['avg_inference_ms'], greaterThan(0));
      expect(metrics['inference_fps'], greaterThan(0));
      
      print('Performance metrics:');
      print('  Avg inference time: ${metrics['avg_inference_ms'].toStringAsFixed(2)}ms');
      print('  Inference FPS: ${metrics['inference_fps'].toStringAsFixed(2)}');
      print('  Audio processing: ${metrics['audio_processing']}');
    });
  });
}

/// Load numpy array from .npy file
/// Simple parser for float32 numpy arrays
Future<Float32List> _loadNumpyArray(String path) async {
  final file = File(path);
  if (!file.existsSync()) {
    throw Exception('Test file not found: $path');
  }
  
  final bytes = await file.readAsBytes();
  
  // Simple .npy parser for float32 arrays
  // Skip numpy header (first 128 bytes typically)
  int headerEnd = 128;
  for (int i = 0; i < bytes.length - 1; i++) {
    if (bytes[i] == 0x0A) {  // newline marks end of header
      headerEnd = i + 1;
      break;
    }
  }
  
  // Read float32 data
  final dataBytes = bytes.sublist(headerEnd);
  final data = Float32List.view(dataBytes.buffer);
  
  return data;
}
