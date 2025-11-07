import 'dart:async';
import 'dart:io';
import 'dart:math' as math;

import 'package:flutter/material.dart';
import 'package:file_picker/file_picker.dart';
import 'package:fl_chart/fl_chart.dart';
import 'package:just_audio/just_audio.dart' as audio;
import 'package:path_provider/path_provider.dart';
import 'package:path/path.dart' as path;
import 'dart:typed_data';

import '../optimized/anomaly_detection_service_optimized.dart';
import '../optimized/audio_processing_service_optimized.dart';
import 'services/live_audio_service_optimized.dart';
import 'pages/mqtt_test_page.dart';
import 'pages/mqtt_simulator_page.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const EdgeSonicApp());
}

class EdgeSonicApp extends StatelessWidget {
  const EdgeSonicApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'EdgeSonic - Optimized',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
        useMaterial3: true,
      ),
      debugShowCheckedModeBanner: false,
      home: const EdgeSonicHomePage(),
    );
  }
}

class EdgeSonicHomePage extends StatefulWidget {
  const EdgeSonicHomePage({super.key});

  @override
  State<EdgeSonicHomePage> createState() => _EdgeSonicHomePageState();
}

class _EdgeSonicHomePageState extends State<EdgeSonicHomePage> {
  final AnomalyDetectionServiceOptimized _detector = AnomalyDetectionServiceOptimized();
  final LiveAudioServiceOptimized _liveAudioService = LiveAudioServiceOptimized();
  StreamSubscription<LiveAudioChunk>? _liveSubscription;

  // Model state
  bool _modelLoaded = false;
  bool _isLoadingModel = false;
  String? _loadError;

  // File processing state
  String? _selectedFilePath;
  String? _selectedFileName;
  bool _isProcessingFile = false;
  double _fileProgress = 0.0;
  List<AnomalyResult>? _fileResults;
  String? _fileError;
  String? _exportPath;

  // Live inference state
  bool _isLiveRunning = false;
  bool _liveInferenceInProgress = false;
  int _liveChunksProcessed = 0;
  AnomalyResult? _latestLiveResult;
  String? _liveError;
  double? _liveRms;
  DateTime? _lastLiveChunkTime;

  @override
  void initState() {
    super.initState();
    _loadModel();
  }

  @override
  void dispose() {
    _liveSubscription?.cancel();
    unawaited(_liveAudioService.stop());
    _liveAudioService.dispose();
    _detector.dispose();
    super.dispose();
  }

  Future<void> _loadModel() async {
    setState(() {
      _isLoadingModel = true;
      _loadError = null;
    });

    try {
      final loaded = await _detector.loadModel();
      if (!mounted) return;
      setState(() {
        _modelLoaded = loaded;
        _isLoadingModel = false;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _modelLoaded = false;
        _isLoadingModel = false;
        _loadError = 'Failed to load model: $e';
      });
    }
  }

  Future<void> _pickAudioFile() async {
    final result = await FilePicker.platform.pickFiles(
      type: FileType.audio,
      allowMultiple: false,
    );

    if (result != null && result.files.single.path != null) {
      setState(() {
        _selectedFilePath = result.files.single.path;
        _selectedFileName = result.files.single.name;
        _fileResults = null;
        _fileError = null;
        _exportPath = null;
      });
    }
  }

  Future<void> _processAudioFile() async {
    if (_selectedFilePath == null || !_modelLoaded) return;

    setState(() {
      _isProcessingFile = true;
      _fileProgress = 0.0;
      _fileError = null;
      _fileResults = null;
      _exportPath = null;
    });

    try {
      // Load audio file
      final audioPlayer = audio.AudioPlayer();
      await audioPlayer.setFilePath(_selectedFilePath!);
      final duration = audioPlayer.duration;

      if (duration == null) {
        throw Exception('Could not determine audio duration');
      }

      // Read audio samples
      final file = File(_selectedFilePath!);
      // For simplicity, we assume WAV format - you may need audio file decoder
      // This is a simplified version - production should use proper audio decoder

      final results = <AnomalyResult>[];

      // Process audio in chunks
      // Note: This is simplified - in production, decode audio properly
      setState(() {
        _fileError = 'Note: Full file processing requires audio decoding library. '
                     'Use live inference for real-time testing.';
        _isProcessingFile = false;
      });

      audioPlayer.dispose();
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _fileError = 'Error processing file: $e';
        _isProcessingFile = false;
      });
    }
  }

  Future<void> _toggleLiveCapture() async {
    if (_isLiveRunning) {
      await _stopLiveCapture();
    } else {
      await _startLiveCapture();
    }
  }

  Future<void> _startLiveCapture() async {
    if (!_modelLoaded) {
      setState(() {
        _liveError = 'Model not loaded';
      });
      return;
    }

    setState(() {
      _liveError = null;
    });

    final result = await _liveAudioService.start();
    if (!mounted) return;

    if (!result.started) {
      setState(() {
        _isLiveRunning = false;
        _liveError = result.error ?? 'Failed to start audio capture';
      });
      return;
    }

    // Clear detector state for fresh inference
    _detector.clearBuffer();
    _detector.resetMetrics();

    await _liveSubscription?.cancel();
    _liveSubscription = _liveAudioService.chunks.listen(
      (chunk) {
        if (!mounted) return;
        _handleLiveChunk(chunk);
      },
      onError: (error) {
        if (!mounted) return;
        setState(() {
          _liveError = 'Audio stream error: $error';
        });
        unawaited(_stopLiveCapture());
      },
    );

    setState(() {
      _isLiveRunning = true;
      _liveChunksProcessed = 0;
      _latestLiveResult = null;
      _liveInferenceInProgress = false;
      _liveRms = null;
      _lastLiveChunkTime = null;
    });
  }

  Future<void> _stopLiveCapture() async {
    await _liveSubscription?.cancel();
    _liveSubscription = null;
    await _liveAudioService.stop();

    if (!mounted) return;
    setState(() {
      _isLiveRunning = false;
      _liveInferenceInProgress = false;
    });
  }

  void _handleLiveChunk(LiveAudioChunk chunk) {
    // Update stats
    setState(() {
      _liveChunksProcessed = _liveAudioService.chunksProcessed;
      _liveRms = chunk.rms;
      _lastLiveChunkTime = chunk.timestamp;
    });

    // Run inference if not already in progress
    if (!_liveInferenceInProgress) {
      _liveInferenceInProgress = true;
      unawaited(_runLiveInference(chunk));
    }
  }

  Future<void> _runLiveInference(LiveAudioChunk chunk) async {
    try {
      final result = await _detector.processAudioChunk(chunk.samples);
      if (!mounted) return;

      setState(() {
        _latestLiveResult = result;
        _liveError = null;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _liveError = 'Inference error: $e';
      });
    } finally {
      _liveInferenceInProgress = false;
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('EdgeSonic - Optimized Inference'),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
      ),
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.all(16),
          children: [
            _buildModelCard(),
            const SizedBox(height: 16),
            _buildLiveAudioCard(),
            const SizedBox(height: 16),
            _buildFileProcessingCard(),
            const SizedBox(height: 16),
            _buildMqttCard(),
            if (_fileResults != null) ...[
              const SizedBox(height: 16),
              _buildResultsCard(),
            ],
          ],
        ),
      ),
    );
  }

  Widget _buildModelCard() {
    final theme = Theme.of(context);
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(
                  _modelLoaded ? Icons.check_circle : Icons.error,
                  color: _modelLoaded ? Colors.green : Colors.red,
                ),
                const SizedBox(width: 8),
                Text(
                  _modelLoaded ? 'Model Ready' : 'Model Not Loaded',
                  style: theme.textTheme.titleMedium,
                ),
                if (_isLoadingModel) ...[
                  const SizedBox(width: 8),
                  const SizedBox(
                    width: 16,
                    height: 16,
                    child: CircularProgressIndicator(strokeWidth: 2),
                  ),
                ],
              ],
            ),
            const SizedBox(height: 8),
            Text(
              'INT8 TFLite Model | Threshold: ${_detector.threshold.toStringAsFixed(4)}',
              style: theme.textTheme.bodySmall,
            ),
            if (_loadError != null) ...[
              const SizedBox(height: 8),
              Text(
                _loadError!,
                style: theme.textTheme.bodySmall?.copyWith(color: Colors.red),
              ),
            ],
            const SizedBox(height: 12),
            FilledButton.icon(
              onPressed: _isLoadingModel ? null : _loadModel,
              icon: const Icon(Icons.refresh),
              label: const Text('Reload Model'),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildLiveAudioCard() {
    final theme = Theme.of(context);
    final statusColor = _isLiveRunning ? Colors.green : Colors.grey;

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(
                  _isLiveRunning ? Icons.mic : Icons.mic_off,
                  color: statusColor,
                ),
                const SizedBox(width: 8),
                Text('Live Microphone Inference', style: theme.textTheme.titleMedium),
              ],
            ),
            const SizedBox(height: 12),

            // Stats
            Wrap(
              spacing: 16,
              runSpacing: 12,
              children: [
                _buildStatItem('Chunks', '$_liveChunksProcessed'),
                if (_liveRms != null)
                  _buildStatItem('RMS', _liveRms!.toStringAsFixed(4)),
                if (_latestLiveResult != null) ...[
                  _buildStatItem(
                    'Score',
                    _latestLiveResult!.smoothedScore.toStringAsFixed(4),
                    color: _latestLiveResult!.isAnomaly ? Colors.red : Colors.green,
                  ),
                  _buildStatItem(
                    'Status',
                    _latestLiveResult!.isAnomaly ? 'ANOMALY' : 'NORMAL',
                    color: _latestLiveResult!.isAnomaly ? Colors.red : Colors.green,
                  ),
                  _buildStatItem(
                    'Latency',
                    '${_latestLiveResult!.processingTimeMs.toStringAsFixed(1)}ms',
                  ),
                ],
              ],
            ),

            const SizedBox(height: 12),
            FilledButton.icon(
              onPressed: _modelLoaded ? _toggleLiveCapture : null,
              icon: Icon(_isLiveRunning ? Icons.stop : Icons.play_arrow),
              label: Text(_isLiveRunning ? 'Stop Capture' : 'Start Live Capture'),
            ),

            if (_liveError != null) ...[
              const SizedBox(height: 12),
              Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: Colors.red.shade50,
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Text(
                  _liveError!,
                  style: TextStyle(color: Colors.red.shade700),
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }

  Widget _buildFileProcessingCard() {
    final theme = Theme.of(context);
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                const Icon(Icons.audio_file),
                const SizedBox(width: 8),
                Text('Audio File Processing', style: theme.textTheme.titleMedium),
              ],
            ),
            const SizedBox(height: 12),

            if (_selectedFileName != null) ...[
              Text('Selected: $_selectedFileName', style: theme.textTheme.bodySmall),
              const SizedBox(height: 8),
            ],

            Row(
              children: [
                ElevatedButton.icon(
                  onPressed: _isProcessingFile ? null : _pickAudioFile,
                  icon: const Icon(Icons.folder_open),
                  label: const Text('Select File'),
                ),
                const SizedBox(width: 8),
                ElevatedButton.icon(
                  onPressed: (_selectedFilePath != null && _modelLoaded && !_isProcessingFile)
                      ? _processAudioFile
                      : null,
                  icon: const Icon(Icons.play_arrow),
                  label: const Text('Process'),
                ),
              ],
            ),

            if (_isProcessingFile) ...[
              const SizedBox(height: 12),
              LinearProgressIndicator(value: _fileProgress),
            ],

            if (_fileError != null) ...[
              const SizedBox(height: 12),
              Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: Colors.orange.shade50,
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Text(
                  _fileError!,
                  style: TextStyle(color: Colors.orange.shade700),
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }

  Widget _buildMqttCard() {
    final theme = Theme.of(context);
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(Icons.router, color: theme.colorScheme.primary),
                const SizedBox(width: 8),
                Text('MQTT Tools', style: theme.textTheme.titleMedium),
              ],
            ),
            const SizedBox(height: 8),
            Text(
              'Test MQTT connectivity and simulate ESP32 telemetry.',
              style: theme.textTheme.bodySmall,
            ),
            const SizedBox(height: 12),
            Row(
              children: [
                Expanded(
                  child: FilledButton.icon(
                    onPressed: () {
                      Navigator.push(
                        context,
                        MaterialPageRoute(builder: (_) => const MqttTestPage()),
                      );
                    },
                    icon: const Icon(Icons.wifi_tethering),
                    label: const Text('MQTT Test'),
                  ),
                ),
                const SizedBox(width: 8),
                Expanded(
                  child: OutlinedButton.icon(
                    onPressed: () {
                      Navigator.push(
                        context,
                        MaterialPageRoute(builder: (_) => const MqttSimulatorPage()),
                      );
                    },
                    icon: const Icon(Icons.sensors),
                    label: const Text('ESP32 Simulator'),
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildResultsCard() {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Results',
              style: Theme.of(context).textTheme.titleMedium,
            ),
            const SizedBox(height: 8),
            // Add results visualization here
            Text('${_fileResults!.length} chunks processed'),
          ],
        ),
      ),
    );
  }

  Widget _buildStatItem(String label, String value, {Color? color}) {
    final theme = Theme.of(context);
    return SizedBox(
      width: 120,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            label,
            style: theme.textTheme.bodySmall?.copyWith(
              color: theme.colorScheme.onSurfaceVariant,
            ),
          ),
          const SizedBox(height: 4),
          Text(
            value,
            style: theme.textTheme.titleSmall?.copyWith(color: color),
          ),
        ],
      ),
    );
  }
}
