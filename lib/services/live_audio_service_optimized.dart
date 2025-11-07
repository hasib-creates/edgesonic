import 'dart:async';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:permission_handler/permission_handler.dart';
import 'package:sound_stream/sound_stream.dart';

import '../../optimized/audio_processing_service_optimized.dart';

class LiveAudioChunk {
  LiveAudioChunk({
    required this.samples,
    required this.modelInput,
    required this.melSpectrogram,
    required this.rms,
    required this.timestamp,
  });

  final Float32List samples;
  final List<List<List<double>>> modelInput;
  final List<List<double>> melSpectrogram;
  final double rms;
  final DateTime timestamp;
}

class LiveAudioStartResult {
  const LiveAudioStartResult({
    required this.started,
    this.error,
  });

  final bool started;
  final String? error;
}

/// Optimized live audio service using the optimized audio processor
class LiveAudioServiceOptimized {
  LiveAudioServiceOptimized({
    RecorderStream? recorder,
    AudioProcessingServiceOptimized? audioProcessor,
  })  : _recorder = recorder ?? RecorderStream(),
        _audioProcessor = audioProcessor ?? AudioProcessingServiceOptimized();

  final RecorderStream _recorder;
  final AudioProcessingServiceOptimized _audioProcessor;

  final StreamController<LiveAudioChunk> _chunkController =
      StreamController<LiveAudioChunk>.broadcast();

  StreamSubscription<List<int>>? _streamSubscription;
  final List<int> _byteBuffer = <int>[];

  bool _isRunning = false;
  int _chunksProcessed = 0;

  static const int _bytesPerSample = 2; // PCM16

  int get _chunkByteCount =>
      AudioProcessingServiceOptimized.samplesPerChunk * _bytesPerSample;

  int get _hopByteCount =>
      AudioProcessingServiceOptimized.hopSamples * _bytesPerSample;

  Stream<LiveAudioChunk> get chunks => _chunkController.stream;

  bool get isRunning => _isRunning;

  int get chunksProcessed => _chunksProcessed;

  Future<LiveAudioStartResult> start() async {
    if (_isRunning) {
      return const LiveAudioStartResult(started: true);
    }

    // Request microphone permission
    final status = await Permission.microphone.request();
    if (!status.isGranted) {
      return const LiveAudioStartResult(
        started: false,
        error: 'Microphone permission denied. Please enable it in settings.',
      );
    }

    // Initialize recorder
    try {
      await _recorder.initialize();
    } catch (error) {
      return LiveAudioStartResult(
        started: false,
        error: 'Failed to initialize microphone: $error',
      );
    }

    _byteBuffer.clear();
    _chunksProcessed = 0;
    _audioProcessor.clearBuffer();

    // Subscribe to audio stream
    _streamSubscription = _recorder.audioStream.listen(
      _onStreamData,
      onError: (Object error) {
        _chunkController.addError(error);
        stop();
      },
      cancelOnError: true,
    );

    // Start recording
    try {
      await _recorder.start();
    } catch (error) {
      await _streamSubscription?.cancel();
      _streamSubscription = null;
      return LiveAudioStartResult(
        started: false,
        error: 'Failed to start recording: $error',
      );
    }

    _isRunning = true;
    print('✅ Live audio capture started');
    print('   Sample rate: ${AudioConfig.sampleRate} Hz');
    print('   Chunk size: ${AudioProcessingServiceOptimized.samplesPerChunk} samples');
    print('   Hop size: ${AudioProcessingServiceOptimized.hopSamples} samples (50% overlap)');

    return const LiveAudioStartResult(started: true);
  }

  Future<void> stop() async {
    if (!_isRunning) {
      return;
    }

    await _streamSubscription?.cancel();
    _streamSubscription = null;
    await _recorder.stop();

    _byteBuffer.clear();
    _isRunning = false;
    print('🛑 Live audio capture stopped');
  }

  void dispose() {
    _chunkController.close();
    _streamSubscription?.cancel();
    unawaited(_recorder.stop());
  }

  void _onStreamData(List<int> data) {
    final Uint8List bytes = data is Uint8List ? data : Uint8List.fromList(data);
    _byteBuffer.addAll(bytes);

    // Process all available chunks with 50% overlap
    while (_byteBuffer.length >= _chunkByteCount) {
      final chunkBytes = Uint8List.fromList(
        _byteBuffer.sublist(0, _chunkByteCount),
      );

      // Convert PCM16 to Float32
      final Float32List samples = _convertPcm16ToFloat(chunkBytes);

      // Process audio through optimized pipeline
      final melSpectrogram = _audioProcessor.audioToMelSpectrogram(samples);
      final modelInput = _audioProcessor.prepareModelInput(melSpectrogram);
      final double rms = _computeRms(samples);

      _chunksProcessed += 1;
      _chunkController.add(
        LiveAudioChunk(
          samples: samples,
          melSpectrogram: melSpectrogram,
          modelInput: modelInput,
          rms: rms,
          timestamp: DateTime.now(),
        ),
      );

      // Remove hop samples for 50% overlap
      if (_byteBuffer.length > _hopByteCount) {
        _byteBuffer.removeRange(0, _hopByteCount);
      } else {
        _byteBuffer.clear();
        break;
      }
    }
  }

  /// Convert PCM16 (16-bit signed integer) to Float32 [-1.0, 1.0]
  Float32List _convertPcm16ToFloat(Uint8List bytes) {
    final sampleCount = bytes.length ~/ _bytesPerSample;
    final Float32List samples = Float32List(sampleCount);

    for (int i = 0; i < sampleCount; i++) {
      final int low = bytes[i * 2];
      final int high = bytes[i * 2 + 1] << 8;
      int sample = high | low;

      // Convert to signed int16
      if (sample & 0x8000 != 0) {
        sample = sample - 0x10000;
      }

      // Normalize to [-1.0, 1.0]
      samples[i] = sample / 32768.0;
    }
    return samples;
  }

  /// Compute RMS (Root Mean Square) energy
  double _computeRms(Float32List samples) {
    double sum = 0;
    for (final sample in samples) {
      sum += sample * sample;
    }
    return samples.isEmpty ? 0.0 : math.sqrt(sum / samples.length);
  }

  /// Get performance metrics
  Map<String, dynamic> getMetrics() {
    return {
      'chunks_processed': _chunksProcessed,
      'is_running': _isRunning,
      'buffer_size': _byteBuffer.length,
      'audio_processor_metrics': _audioProcessor.getPerformanceMetrics(),
    };
  }

  /// Reset metrics
  void resetMetrics() {
    _chunksProcessed = 0;
    _audioProcessor.resetMetrics();
  }
}
