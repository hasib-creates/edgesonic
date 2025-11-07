import 'dart:async';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:permission_handler/permission_handler.dart';
import 'package:sound_stream/sound_stream.dart';

import 'audio_processing_service.dart';

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

  LiveAudioStartResult copyWith({
    bool? started,
    String? error,
  }) {
    return LiveAudioStartResult(
      started: started ?? this.started,
      error: error ?? this.error,
    );
  }
}

class LiveAudioService {
  LiveAudioService({
    RecorderStream? recorder,
    AudioProcessingService? audioProcessor,
  })  : _recorder = recorder ?? RecorderStream(),
        _audioProcessor = audioProcessor ?? AudioProcessingService();

  final RecorderStream _recorder;
  final AudioProcessingService _audioProcessor;

  final StreamController<LiveAudioChunk> _chunkController =
      StreamController<LiveAudioChunk>.broadcast();

  StreamSubscription<List<int>>? _streamSubscription;
  final List<int> _byteBuffer = <int>[];

  bool _isRunning = false;
  int _chunksProcessed = 0;

  static const int _bytesPerSample = 2; // PCM16

  int get _chunkByteCount =>
      AudioProcessingService.samplesPerChunk * _bytesPerSample;

  int get _hopByteCount =>
      AudioProcessingService.hopSamples * _bytesPerSample;

  Stream<LiveAudioChunk> get chunks => _chunkController.stream;

  bool get isRunning => _isRunning;

  int get chunksProcessed => _chunksProcessed;

  Future<LiveAudioStartResult> start() async {
    if (_isRunning) {
      return const LiveAudioStartResult(started: true);
    }

    final status = await Permission.microphone.request();
    if (!status.isGranted) {
      return const LiveAudioStartResult(
        started: false,
        error: 'Microphone permission is required. Enable it in Android settings and try again.',
      );
    }

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

    _streamSubscription = _recorder.audioStream.listen(
      _onStreamData,
      onError: (Object error) {
        _chunkController.addError(error);
        stop();
      },
      cancelOnError: true,
    );

    try {
      await _recorder.start();
    } catch (error) {
      await _streamSubscription?.cancel();
      _streamSubscription = null;
      return LiveAudioStartResult(
        started: false,
        error: 'Failed to start microphone: $error',
      );
    }

    _isRunning = true;
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
  }

  void dispose() {
    _chunkController.close();
    _streamSubscription?.cancel();
    unawaited(_recorder.stop());
  }

  void _onStreamData(List<int> data) {
    final Uint8List bytes = data is Uint8List ? data : Uint8List.fromList(data);
    _byteBuffer.addAll(bytes);

    while (_byteBuffer.length >= _chunkByteCount) {
      final chunkBytes = Uint8List.fromList(
        _byteBuffer.sublist(0, _chunkByteCount),
      );

      final Float32List samples = _convertPcm16ToFloat(chunkBytes);
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

      if (_byteBuffer.length > _hopByteCount) {
        _byteBuffer.removeRange(0, _hopByteCount);
      } else {
        _byteBuffer.clear();
        break;
      }
    }
  }

  Float32List _convertPcm16ToFloat(Uint8List bytes) {
    final sampleCount = bytes.length ~/ _bytesPerSample;
    final Float32List samples = Float32List(sampleCount);

    for (int i = 0; i < sampleCount; i++) {
      final int low = bytes[i * 2];
      final int high = bytes[i * 2 + 1] << 8;
      int sample = high | low;

      if (sample & 0x8000 != 0) {
        sample = sample - 0x10000;
      }

      samples[i] = sample / 32768.0;
    }
    return samples;
  }

  double _computeRms(Float32List samples) {
    double sum = 0;
    for (final sample in samples) {
      sum += sample * sample;
    }
    return samples.isEmpty ? 0.0 : math.sqrt(sum / samples.length);
  }
}
