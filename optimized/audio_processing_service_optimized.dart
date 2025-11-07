import 'dart:typed_data';
import 'dart:math' as math;
import 'package:fftea/fftea.dart';

class AudioConfig {
  static const int sampleRate = 16000;
  static const int nFft = 512;
  static const int hopLength = 256;
  static const int numMelBins = 16;
  static const int targetLength = 128;
  static const double normMean = -5.0;
  static const double normStd = 4.5;
}

/// Optimized audio processing service for real-time anomaly detection
/// Key optimizations:
/// 1. Pre-computed Hanning window
/// 2. Reusable FFT buffers
/// 3. Circular buffer for streaming
/// 4. Efficient mel filterbank application
class AudioProcessingServiceOptimized {
  // Pre-computed components
  late List<List<double>> melFilterbank;
  late List<double> hanningWindow;
  late FFT fft;
  
  // Reusable buffers (avoid repeated allocations)
  late Float64List _fftBuffer;
  late List<double> _powerSpectrum;
  
  // Circular buffer for streaming
  final List<double> _audioBuffer = [];
  static const int _maxBufferSize = 100000; // ~6 seconds at 16kHz
  
  // Performance metrics
  int _processedChunks = 0;
  double _totalProcessingTimeMs = 0.0;
  
  static int get samplesPerChunk =>
      (AudioConfig.targetLength - 1) * AudioConfig.hopLength + AudioConfig.nFft;

  static int get hopSamples => samplesPerChunk ~/ 2;

  AudioProcessingServiceOptimized() {
    _initialize();
  }

  void _initialize() {
    // Pre-compute mel filterbank
    melFilterbank = _createMelFilterbank();
    
    // Pre-compute Hanning window
    hanningWindow = List<double>.generate(AudioConfig.nFft, (i) {
      return 0.5 - 0.5 * math.cos(2 * math.pi * i / (AudioConfig.nFft - 1));
    });
    
    // Initialize FFT
    fft = FFT(AudioConfig.nFft);
    
    // Allocate reusable buffers
    _fftBuffer = Float64List(AudioConfig.nFft);
    _powerSpectrum = List<double>.filled(AudioConfig.nFft ~/ 2 + 1, 0.0);
    
    print('AudioProcessingServiceOptimized initialized');
    print('  Samples per chunk: $samplesPerChunk (~${(samplesPerChunk/AudioConfig.sampleRate).toStringAsFixed(2)}s)');
    print('  Hop samples: $hopSamples (~${(hopSamples/AudioConfig.sampleRate).toStringAsFixed(2)}s overlap)');
  }

  /// Create Mel filterbank matrix (optimized)
  List<List<double>> _createMelFilterbank() {
    final filterbank = List<List<double>>.generate(
      AudioConfig.numMelBins,
      (_) => List<double>.filled(AudioConfig.nFft ~/ 2 + 1, 0.0),
    );
    
    // Hz to Mel conversion
    double hzToMel(double hz) => 2595.0 * math.log(1.0 + hz / 700.0) / math.ln10;
    double melToHz(double mel) => 700.0 * (math.pow(10.0, mel / 2595.0) - 1.0);
    
    final melMax = hzToMel(AudioConfig.sampleRate / 2.0);
    final melPoints = List<double>.generate(
      AudioConfig.numMelBins + 2,
      (i) => i * melMax / (AudioConfig.numMelBins + 1),
    );
    
    final hzPoints = melPoints.map(melToHz).toList();
    final binPoints = hzPoints.map((hz) {
      return (AudioConfig.nFft + 1) * hz / AudioConfig.sampleRate;
    }).toList();
    
    for (int i = 0; i < AudioConfig.numMelBins; i++) {
      final start = binPoints[i].floor();
      final peak = binPoints[i + 1].floor();
      final end = binPoints[i + 2].floor();
      
      // Rising edge
      for (int j = start; j < peak && j < filterbank[i].length; j++) {
        final denominator = math.max(peak - start, 1);
        filterbank[i][j] = (j - start) / denominator;
      }
      
      // Falling edge
      for (int j = peak; j < end && j < filterbank[i].length; j++) {
        final denominator = math.max(end - peak, 1);
        filterbank[i][j] = (end - j) / denominator;
      }
    }
    
    return filterbank;
  }

  /// Optimized: Compute single frame of mel spectrogram
  /// This is the core hot-path function - heavily optimized
  List<double> _computeMelFrame(Float32List audioFrame, int start) {
    // Apply window (reuse buffer)
    for (int i = 0; i < AudioConfig.nFft; i++) {
      _fftBuffer[i] = audioFrame[start + i] * hanningWindow[i];
    }
    
    // Compute FFT (reuse FFT object)
    final spectrum = fft.realFft(_fftBuffer);
    
    // Convert to power spectrum (reuse buffer)
    const halfSize = AudioConfig.nFft ~/ 2 + 1;
    for (int i = 0; i < halfSize && i < spectrum.length; i++) {
      final real = spectrum[i].x;
      final imag = spectrum[i].y;
      _powerSpectrum[i] = real * real + imag * imag;
    }
    
    // Apply Mel filterbank
    final melFrame = List<double>.filled(AudioConfig.numMelBins, 0.0);
    for (int melBin = 0; melBin < AudioConfig.numMelBins; melBin++) {
      double melValue = 0.0;
      for (int i = 0; i < _powerSpectrum.length && i < melFilterbank[melBin].length; i++) {
        melValue += _powerSpectrum[i] * melFilterbank[melBin][i];
      }
      // Log scale + normalization
      melFrame[melBin] = (math.log(melValue + 1e-6) - AudioConfig.normMean) / AudioConfig.normStd;
    }
    
    return melFrame;
  }

  /// Convert audio samples to Mel spectrogram (optimized for streaming)
  List<List<double>> audioToMelSpectrogram(Float32List audioData) {
    final stopwatch = Stopwatch()..start();
    
    final numFrames = ((audioData.length - AudioConfig.nFft) ~/ AudioConfig.hopLength) + 1;
    final melSpectrogram = List<List<double>>.generate(
      numFrames,
      (_) => List<double>.filled(AudioConfig.numMelBins, 0.0),
    );
    
    // Process each frame
    for (int frameIdx = 0; frameIdx < numFrames; frameIdx++) {
      final start = frameIdx * AudioConfig.hopLength;
      if (start + AudioConfig.nFft > audioData.length) break;
      
      melSpectrogram[frameIdx] = _computeMelFrame(audioData, start);
    }
    
    stopwatch.stop();
    _processedChunks++;
    _totalProcessingTimeMs += stopwatch.elapsedMicroseconds / 1000.0;
    
    // Transpose to [numMelBins, numFrames] format
    return _transposeSpectrogram(melSpectrogram);
  }

  /// Transpose spectrogram from [frames, mels] to [mels, frames]
  List<List<double>> _transposeSpectrogram(List<List<double>> melSpec) {
    final transposed = List<List<double>>.generate(
      AudioConfig.numMelBins,
      (melIdx) => List<double>.generate(
        melSpec.length,
        (frameIdx) => melSpec[frameIdx][melIdx],
      ),
    );
    return transposed;
  }

  /// Streaming: Add new audio samples and extract ready chunks
  List<Float32List> addAudioSamples(Float32List newSamples) {
    // Add to circular buffer
    for (int i = 0; i < newSamples.length; i++) {
      _audioBuffer.add(newSamples[i]);
    }
    
    // Limit buffer size
    if (_audioBuffer.length > _maxBufferSize) {
      _audioBuffer.removeRange(0, _audioBuffer.length - _maxBufferSize);
    }
    
    // Extract complete chunks
    final chunks = <Float32List>[];
    while (_audioBuffer.length >= samplesPerChunk) {
      final chunk = Float32List.fromList(_audioBuffer.sublist(0, samplesPerChunk));
      chunks.add(chunk);
      
      // Remove processed samples (50% overlap)
      _audioBuffer.removeRange(0, hopSamples);
    }
    
    return chunks;
  }

  /// Clear the streaming buffer
  void clearBuffer() {
    _audioBuffer.clear();
  }

  /// Prepare input for the model (shape: [1, 16, 128])
  List<List<List<double>>> prepareModelInput(List<List<double>> melSpectrogram) {
    // Ensure we have exactly 128 frames
    final processedMel = List<List<double>>.generate(AudioConfig.numMelBins, (i) {
      final row = List<double>.filled(AudioConfig.targetLength, 0.0);
      final sourceLength = math.min(melSpectrogram[i].length, AudioConfig.targetLength);
      for (int j = 0; j < sourceLength; j++) {
        row[j] = melSpectrogram[i][j];
      }
      return row;
    });
    
    // Return in batch format [1, 16, 128]
    return [processedMel];
  }

  /// Get performance metrics
  Map<String, dynamic> getPerformanceMetrics() {
    final avgTimeMs = _processedChunks > 0 ? _totalProcessingTimeMs / _processedChunks : 0.0;
    return {
      'processed_chunks': _processedChunks,
      'avg_time_ms': avgTimeMs,
      'total_time_ms': _totalProcessingTimeMs,
      'fps': avgTimeMs > 0 ? 1000.0 / avgTimeMs : 0.0,
    };
  }

  /// Reset performance metrics
  void resetMetrics() {
    _processedChunks = 0;
    _totalProcessingTimeMs = 0.0;
  }
}
