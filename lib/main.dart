import 'dart:async';
import 'dart:io';
import 'dart:math' as math;

import 'package:flutter/material.dart';
import 'package:file_picker/file_picker.dart';
import 'package:fl_chart/fl_chart.dart';
import 'services/anomaly_detection_service.dart';
import 'services/live_audio_service.dart';
import 'services/audio_processing_service.dart';
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
      title: 'EdgeSonic Anomaly Detection',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.teal),
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
  final AnomalyDetectionService _detectionService = AnomalyDetectionService();
  final LiveAudioService _liveAudioService = LiveAudioService();
  StreamSubscription<LiveAudioChunk>? _liveSubscription;

  bool _modelLoaded = false;
  bool _isLoadingModel = false;
  bool _isProcessing = false;
  String _modelSourceLabel = 'Asset';
  String? _customModelPath;
  bool _metadataLoaded = false;
  String? _metadataStatus;
  String? _selectedFilePath;
  String? _selectedFileName;
  double _processingProgress = 0.0;
  List<AnomalyDetectionResult>? _results;
  String? _errorMessage;
  String? _lastExportPath;
  String? _exportError;
  bool _isLiveRunning = false;
  bool _liveInferenceInProgress = false;
  int _liveChunksProcessed = 0;
  double? _liveRms;
  double? _liveMelMean;
  double? _liveMelStd;
  DateTime? _lastLiveChunkTime;
  String? _liveError;
  AnomalyDetectionResult? _latestLiveResult;

  @override
  void initState() {
    super.initState();
    _loadModel();
  }

  Widget _buildLiveAudioCard(BuildContext context) {
    final theme = Theme.of(context);
    final statusColor = _isLiveRunning ? Colors.green : Colors.grey;
    final statusText =
        _isLiveRunning ? 'Streaming from microphone' : 'Idle';
    final lastChunkTime = _lastLiveChunkTime != null
        ? TimeOfDay.fromDateTime(_lastLiveChunkTime!).format(context)
        : '—';
    final liveResult = _latestLiveResult;
    final bool hasLiveResult = liveResult != null;
    final String anomalyText;
    final Color anomalyColor;
    if (!hasLiveResult) {
      anomalyText = 'Awaiting inference';
      anomalyColor = theme.colorScheme.onSurfaceVariant;
    } else if (liveResult!.isAnomaly) {
      anomalyText = 'Anomaly detected';
      anomalyColor = Colors.red;
    } else {
      anomalyText = 'Normal';
      anomalyColor = Colors.green;
    }

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16.0),
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
                Text(
                  'Live Microphone (beta)',
                  style: theme.textTheme.titleMedium,
                ),
              ],
            ),
            const SizedBox(height: 8),
            Text(
              statusText,
              style: theme.textTheme.bodySmall,
            ),
            if (hasLiveResult) ...[
              const SizedBox(height: 4),
              Text(
                anomalyText,
                style: theme.textTheme.bodySmall?.copyWith(color: anomalyColor),
              ),
            ],
            const SizedBox(height: 12),
            Wrap(
              spacing: 16,
              runSpacing: 12,
              children: [
                _buildLiveStat(context, 'Windows', '$_liveChunksProcessed'),
                _buildLiveStat(
                  context,
                  'RMS',
                  _liveRms != null ? _liveRms!.toStringAsFixed(4) : '—',
                ),
                _buildLiveStat(
                  context,
                  'Mel mean',
                  _liveMelMean != null ? _liveMelMean!.toStringAsFixed(4) : '—',
                ),
                _buildLiveStat(
                  context,
                  'Mel std',
                  _liveMelStd != null ? _liveMelStd!.toStringAsFixed(4) : '—',
                ),
                _buildLiveStat(
                  context,
                  'Last chunk',
                  lastChunkTime,
                ),
                if (hasLiveResult) ...[
                  _buildLiveStat(
                    context,
                    'MSE',
                    liveResult!.rawScore.toStringAsFixed(4),
                    subtitle: 'thr ${liveResult.threshold.toStringAsFixed(4)}',
                    valueColor: liveResult.isAnomaly ? Colors.red : null,
                  ),
                  if (_metadataLoaded && !liveResult.svddScore.isNaN)
                    _buildLiveStat(
                      context,
                      'SVDD',
                      liveResult.svddScore.toStringAsFixed(4),
                      subtitle: 'thr ${liveResult.svddThreshold.toStringAsFixed(4)}',
                      valueColor: liveResult.isSvddAnomaly ? Colors.red : null,
                    ),
                  _buildLiveStat(
                    context,
                    'Latency',
                    liveResult.inferenceDuration != null
                        ? '${liveResult.inferenceDuration!.inMilliseconds} ms'
                        : '—',
                  ),
                ],
              ],
            ),
            const SizedBox(height: 12),
            Row(
              children: [
                FilledButton.icon(
                  onPressed: () async {
                    await _toggleLiveCapture();
                  },
                  icon: Icon(_isLiveRunning ? Icons.stop : Icons.play_arrow),
                  label: Text(_isLiveRunning ? 'Stop Capture' : 'Start Capture'),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: Text(
                    'Captures 16 kHz PCM and prepares Mel windows (16×128) for the TFLite model.',
                    style: theme.textTheme.bodySmall,
                  ),
                ),
              ],
            ),
            if (_liveError != null) ...[
              const SizedBox(height: 12),
              DecoratedBox(
                decoration: BoxDecoration(
                  color: Colors.red.shade50,
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Padding(
                  padding: const EdgeInsets.all(12),
                  child: Text(
                    _liveError!,
                    style: TextStyle(color: Colors.red.shade700),
                  ),
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }

  Widget _buildLiveStat(
    BuildContext context,
    String label,
    String value, {
    String? subtitle,
    Color? valueColor,
  }) {
    final theme = Theme.of(context);
    return SizedBox(
      width: 140,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            label,
            style: theme.textTheme.bodySmall?.copyWith(
              color: theme.colorScheme.onSurface.withOpacity(0.6),
            ),
          ),
          const SizedBox(height: 4),
          Text(
            value,
            style: theme.textTheme.titleMedium?.copyWith(color: valueColor),
          ),
          if (subtitle != null) ...[
            const SizedBox(height: 2),
            Text(
              subtitle,
              style: theme.textTheme.bodySmall?.copyWith(
                color: theme.colorScheme.onSurfaceVariant,
              ),
            ),
          ],
        ],
      ),
    );
  }

  Widget _buildMqttToolsCard(BuildContext context) {
    final theme = Theme.of(context);
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(Icons.router, color: theme.colorScheme.primary),
                const SizedBox(width: 8),
                Text(
                  'MQTT Tools',
                  style: theme.textTheme.titleMedium,
                ),
              ],
            ),
            const SizedBox(height: 8),
            Text(
              'Connect to your broker, send telemetry, or emulate the ESP32 payloads.',
              style: theme.textTheme.bodySmall,
            ),
            const SizedBox(height: 16),
            Row(
              children: [
                Expanded(
                  child: FilledButton.icon(
                    onPressed: () {
                      Navigator.of(context).push(
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
                      Navigator.of(context).push(
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

  @override
  void dispose() {
    _liveSubscription?.cancel();
    unawaited(_liveAudioService.stop());
    _liveAudioService.dispose();
    _detectionService.dispose();
    super.dispose();
  }

  Future<void> _loadModel({String? filePath}) async {
    setState(() {
      _isLoadingModel = true;
      _errorMessage = null;
    });

    final bool loaded = filePath == null
        ? await _detectionService.loadModel()
        : await _detectionService.loadModelFromFile(filePath);

    bool metadataLoaded = false;
    String? metadataStatus;

    if (loaded) {
      if (filePath == null) {
        metadataLoaded =
            await _detectionService.loadMetadataFromAsset('assets/models/model_metadata.json');
        metadataLoaded = metadataLoaded && _detectionService.hasSvddCenter;
        metadataStatus = metadataLoaded
            ? 'SVDD center length: ${_detectionService.svddCenterLength}'
            : 'SVDD scoring disabled (no metadata found). MSE scoring still works.';
      } else {
        _detectionService.setMetadata(ModelMetadata.defaults());
        metadataLoaded = false;
        metadataStatus = 'Custom model loaded. SVDD scoring disabled unless metadata is provided.';
      }
    } else {
      metadataStatus = null;
    }

    if (!mounted) {
      return;
    }
    setState(() {
      _isLoadingModel = false;
      _modelLoaded = loaded;
      _modelSourceLabel = _detectionService.modelSource;
      _customModelPath = filePath;
      _metadataLoaded = metadataLoaded;
      _metadataStatus = metadataStatus;

      if (!loaded) {
        final details = _detectionService.lastError ?? 'Unknown error';
        _errorMessage =
            'Failed to load model. Confirm the TensorFlow Lite runtime is available.\n$details';
      }
    });
  }

  Future<void> _pickAudioFile() async {
    final result = await FilePicker.platform.pickFiles(
      type: FileType.audio,
    );

    if (result != null && result.files.single.path != null) {
      setState(() {
        _selectedFilePath = result.files.single.path;
        _selectedFileName = result.files.single.name;
        _results = null;
      });
    }
  }

  Future<void> _pickModelFile() async {
    final result = await FilePicker.platform.pickFiles(
      type: FileType.custom,
      allowedExtensions: const ['tflite'],
    );
    final path = result?.files.single.path;
    if (path == null) {
      return;
    }
    await _loadModel(filePath: path);
  }

  Future<void> _pickMetadataFile() async {
    final result = await FilePicker.platform.pickFiles(
      type: FileType.custom,
      allowedExtensions: const ['json'],
    );
    final path = result?.files.single.path;
    if (path == null) {
      return;
    }
    final success = await _detectionService.loadMetadataFromFile(path);
    if (!mounted) {
      return;
    }
    setState(() {
      _metadataLoaded = success && _detectionService.hasSvddCenter;
      _metadataStatus = _metadataLoaded
          ? 'SVDD center length: ${_detectionService.svddCenterLength}'
          : 'Metadata load failed or missing svdd_center. SVDD scoring disabled.';
    });
  }

  Future<void> _processAudioFile() async {
    if (_selectedFilePath == null || !_modelLoaded) return;

    setState(() {
      _isProcessing = true;
      _processingProgress = 0.0;
      _errorMessage = null;
      _exportError = null;
      _lastExportPath = null;
    });

    try {
      final results = await _detectionService.processAudioFile(
        _selectedFilePath!,
        onProgress: (progress) {
          setState(() {
            _processingProgress = progress;
          });
        },
      );

      File? exportFile;
      String? exportError;
      try {
        final hopSeconds = AudioProcessingService.hopSamples / AudioConfig.sampleRate;
        exportFile = await _detectionService.exportResultsToCsv(
          results: results,
          inputFileName: _selectedFileName ?? 'audio_clip',
          hopSeconds: hopSeconds,
        );
      } catch (e) {
        exportError = 'Failed to export CSV: $e';
      }

      setState(() {
        _results = results;
        _isProcessing = false;
        _lastExportPath = exportFile?.path;
        _exportError = exportError;
      });
    } catch (e) {
      setState(() {
        _errorMessage = 'Error processing audio: $e';
        _isProcessing = false;
        _lastExportPath = null;
        _exportError = null;
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
    setState(() {
      _liveError = null;
    });

    final result = await _liveAudioService.start();
    if (!mounted) {
      return;
    }

    if (!result.started) {
      setState(() {
        _isLiveRunning = false;
        _liveError = result.error ?? 'Unable to start live audio capture.';
      });
      return;
    }

    await _liveSubscription?.cancel();
    _liveSubscription = _liveAudioService.chunks.listen(
      (chunk) {
        if (!mounted) return;
        _handleLiveChunk(chunk);
      },
      onError: (Object error) {
        if (!mounted) return;
        setState(() {
          _liveError = 'Live audio error: $error';
        });
        unawaited(_stopLiveCapture());
      },
    );

    setState(() {
      _isLiveRunning = true;
      _liveChunksProcessed = 0;
      _liveRms = null;
      _liveMelMean = null;
      _liveMelStd = null;
      _lastLiveChunkTime = null;
      _latestLiveResult = null;
      _liveInferenceInProgress = false;
    });
  }

  Future<void> _stopLiveCapture() async {
    await _liveSubscription?.cancel();
    _liveSubscription = null;
    await _liveAudioService.stop();
    if (!mounted) {
      return;
    }
    setState(() {
      _isLiveRunning = false;
      _liveInferenceInProgress = false;
      _latestLiveResult = null;
    });
  }

  void _updateLiveStats(LiveAudioChunk chunk) {
    double sum = 0.0;
    double sumSq = 0.0;
    int count = 0;

    for (final row in chunk.melSpectrogram) {
      for (final value in row) {
        sum += value;
        sumSq += value * value;
        count += 1;
      }
    }

    final mean = count == 0 ? 0.0 : sum / count;
    final variance = count == 0 ? 0.0 : (sumSq / count) - (mean * mean);
    final stdDev = variance <= 0 ? 0.0 : math.sqrt(variance);

    setState(() {
      _liveChunksProcessed = _liveAudioService.chunksProcessed;
      _liveRms = chunk.rms;
      _liveMelMean = mean;
      _liveMelStd = stdDev;
      _lastLiveChunkTime = chunk.timestamp;
    });
  }

  void _handleLiveChunk(LiveAudioChunk chunk) {
    _updateLiveStats(chunk);
    if (!_modelLoaded || _liveInferenceInProgress) {
      return;
    }
    _liveInferenceInProgress = true;
    unawaited(_runLiveInference(chunk));
  }

  Future<void> _runLiveInference(LiveAudioChunk chunk) async {
    try {
      final result = await _detectionService.detectAnomalyFromPreparedInput(
        chunk.modelInput,
        timestamp: chunk.timestamp,
      );
      if (!mounted) return;
      setState(() {
        _latestLiveResult = result;
        _liveError = null;
      });
    } catch (error) {
      if (!mounted) return;
      setState(() {
        _liveError = 'Live inference error: $error';
      });
    } finally {
      _liveInferenceInProgress = false;
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('EdgeSonic Anomaly Detection'),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
      ),
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(16.0),
          child: ListView(
            children: [
              // Model Status Card
              Card(
                child: Padding(
                  padding: const EdgeInsets.all(16.0),
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
                            _modelLoaded
                                ? 'Model Loaded Successfully'
                                : 'Model Not Loaded',
                            style: Theme.of(context).textTheme.titleMedium,
                          ),
                          if (_isLoadingModel) ...[
                            const SizedBox(width: 8),
                            const SizedBox(
                              height: 16,
                              width: 16,
                              child: CircularProgressIndicator(strokeWidth: 2),
                            ),
                          ],
                        ],
                      ),
                      const SizedBox(height: 8),
                      Text(
                        'Source: $_modelSourceLabel',
                        style: Theme.of(context).textTheme.bodySmall,
                      ),
                      Text(
                        'Model file: ${_customModelPath ?? 'assets/models/tcn_model_int8.tflite'}',
                        style: Theme.of(context).textTheme.bodySmall,
                      ),
                      Text(
                        'MSE threshold: ${_detectionService.threshold.toStringAsFixed(4)}',
                        style: Theme.of(context).textTheme.bodySmall,
                      ),
                      if (_metadataLoaded)
                        Text(
                          'SVDD threshold: ${_detectionService.svddThreshold.toStringAsFixed(4)}',
                          style: Theme.of(context).textTheme.bodySmall,
                        ),
                      if (_metadataStatus != null) ...[
                        const SizedBox(height: 8),
                        Text(
                          _metadataStatus!,
                          style: Theme.of(context).textTheme.bodySmall,
                        ),
                      ],
                      const SizedBox(height: 12),
                      Wrap(
                        spacing: 8,
                        runSpacing: 8,
                        children: [
                          FilledButton.icon(
                            onPressed: _isLoadingModel ? null : () => _loadModel(),
                            icon: const Icon(Icons.refresh),
                            label: const Text('Reload asset'),
                          ),
                          OutlinedButton.icon(
                            onPressed: _isLoadingModel ? null : _pickModelFile,
                            icon: const Icon(Icons.file_open),
                            label: const Text('Load .tflite'),
                          ),
                          OutlinedButton.icon(
                            onPressed: (!_modelLoaded || _isLoadingModel) ? null : _pickMetadataFile,
                            icon: const Icon(Icons.data_object),
                            label: const Text('Load metadata'),
                          ),
                        ],
                      ),
                    ],
                  ),
                ),
              ),

              const SizedBox(height: 16),

              _buildLiveAudioCard(context),

              const SizedBox(height: 16),

              // File Selection
              Card(
                child: Padding(
                  padding: const EdgeInsets.all(16.0),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'Audio File',
                        style: Theme.of(context).textTheme.titleMedium,
                      ),
                      const SizedBox(height: 8),
                      if (_selectedFileName != null) ...[
                        Row(
                          children: [
                            const Icon(Icons.audio_file),
                            const SizedBox(width: 8),
                            Expanded(
                              child: Text(
                                _selectedFileName!,
                                overflow: TextOverflow.ellipsis,
                              ),
                            ),
                          ],
                        ),
                        const SizedBox(height: 8),
                      ],
                      Row(
                        children: [
                          ElevatedButton.icon(
                            onPressed: _isProcessing ? null : _pickAudioFile,
                            icon: const Icon(Icons.folder_open),
                            label: const Text('Select Audio File'),
                          ),
                          const SizedBox(width: 8),
                          ElevatedButton.icon(
                            onPressed: (_selectedFilePath != null &&
                                _modelLoaded &&
                                !_isProcessing)
                                ? _processAudioFile
                                : null,
                            icon: const Icon(Icons.play_arrow),
                            label: const Text('Process'),
                          ),
                        ],
                      ),
                    ],
                  ),
                ),
              ),

              // Processing Progress
              if (_isProcessing) ...[
                const SizedBox(height: 16),
                Card(
                  child: Padding(
                    padding: const EdgeInsets.all(16.0),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          'Processing: ${(_processingProgress * 100).toStringAsFixed(1)}%',
                          style: Theme.of(context).textTheme.titleMedium,
                        ),
                        const SizedBox(height: 8),
                        LinearProgressIndicator(value: _processingProgress),
                      ],
                    ),
                  ),
                ),
              ],

              // Results
              if (_results != null && _results!.isNotEmpty) ...[
                const SizedBox(height: 16),
                SizedBox(
                  height: 320,
                  child: Card(
                    child: Padding(
                      padding: const EdgeInsets.all(16.0),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            'Anomaly Detection Results',
                            style: Theme.of(context).textTheme.titleMedium,
                          ),
                          const SizedBox(height: 8),
                          _buildStatistics(),
                          const SizedBox(height: 16),
                          Expanded(
                            child: _buildChart(),
                          ),
                        ],
                      ),
                    ),
                  ),
                ),
              ],

              if (_lastExportPath != null || _exportError != null) ...[
                const SizedBox(height: 16),
                _buildExportStatusCard(context),
              ],

              // Error Message
              if (_errorMessage != null) ...[
                const SizedBox(height: 16),
                Card(
                  color: Colors.red.shade50,
                  child: Padding(
                    padding: const EdgeInsets.all(16.0),
                    child: Text(
                      _errorMessage!,
                      style: TextStyle(color: Colors.red.shade900),
                    ),
                  ),
                ),
              ],

              const SizedBox(height: 16),

              _buildMqttToolsCard(context),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildStatistics() {
    if (_results == null || _results!.isEmpty) return Container();

    final scores = _results!.map((r) => r.smoothedScore).toList();
    final minScore = scores.reduce((a, b) => a < b ? a : b);
    final maxScore = scores.reduce((a, b) => a > b ? a : b);
    final avgScore = scores.reduce((a, b) => a + b) / scores.length;
    final anomalyCount = _results!.where((r) => r.isAnomaly).length;
    final svddValues = _results!
        .map((r) => r.svddScore)
        .where((value) => !value.isNaN)
        .toList();
    double? minSvdd;
    double? maxSvdd;
    double? avgSvdd;
    if (svddValues.isNotEmpty) {
      minSvdd = svddValues.reduce((a, b) => a < b ? a : b);
      maxSvdd = svddValues.reduce((a, b) => a > b ? a : b);
      avgSvdd = svddValues.reduce((a, b) => a + b) / svddValues.length;
    }

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text('Min Score: ${minScore.toStringAsFixed(6)}'),
        Text('Max Score: ${maxScore.toStringAsFixed(6)}'),
        Text('Avg Score: ${avgScore.toStringAsFixed(6)}'),
        Text('Anomalies Detected: $anomalyCount / ${_results!.length}'),
        if (minSvdd != null) ...[
          Text('SVDD Min: ${minSvdd.toStringAsFixed(4)}'),
          Text('SVDD Max: ${maxSvdd!.toStringAsFixed(4)}'),
          Text('SVDD Avg: ${avgSvdd!.toStringAsFixed(4)}'),
        ],
      ],
    );
  }

  Widget _buildExportStatusCard(BuildContext context) {
    final theme = Theme.of(context);
    final success = _lastExportPath != null;
    final statusIcon = success ? Icons.check_circle : Icons.info;
    final statusColor = success ? Colors.green : Colors.orange;
    final statusText = success
        ? 'CSV export saved here:'
        : 'CSV export unavailable.';

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(statusIcon, color: statusColor),
                const SizedBox(width: 8),
                Text(
                  statusText,
                  style: theme.textTheme.titleMedium,
                ),
              ],
            ),
            const SizedBox(height: 8),
            if (success)
              SelectableText(
                _lastExportPath!,
                style: theme.textTheme.bodySmall,
              ),
            if (_exportError != null) ...[
              SelectableText(
                _exportError!,
                style: theme.textTheme.bodySmall?.copyWith(color: Colors.orange.shade700),
              ),
            ],
            if (success) ...[
              const SizedBox(height: 8),
              Text(
                'Pull with adb: adb pull "${_lastExportPath!}" ./android_results/',
                style: theme.textTheme.bodySmall?.copyWith(
                  color: theme.colorScheme.onSurfaceVariant,
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }

  Widget _buildChart() {
    if (_results == null || _results!.isEmpty) {
      return const Center(child: Text('No data to display'));
    }

    final spots = _results!
        .asMap()
        .entries
        .map((entry) => FlSpot(
      entry.value.timeSeconds ?? entry.key.toDouble(),
      entry.value.smoothedScore,
    ))
        .toList();

    final threshold = _detectionService.threshold;

    return LineChart(
      LineChartData(
        gridData: const FlGridData(show: true),
        titlesData: FlTitlesData(
          leftTitles: AxisTitles(
            sideTitles: SideTitles(
              showTitles: true,
              reservedSize: 40,
              getTitlesWidget: (value, meta) {
                return Text(value.toStringAsFixed(3));
              },
            ),
          ),
          bottomTitles: AxisTitles(
            axisNameWidget: const Text('Time (s)'),
            sideTitles: SideTitles(
              showTitles: true,
              reservedSize: 30,
              getTitlesWidget: (value, meta) {
                return Text(value.toStringAsFixed(1));
              },
            ),
          ),
          topTitles: const AxisTitles(
            sideTitles: SideTitles(showTitles: false),
          ),
          rightTitles: const AxisTitles(
            sideTitles: SideTitles(showTitles: false),
          ),
        ),
        borderData: FlBorderData(show: true),
        lineBarsData: [
          // Anomaly scores line
          LineChartBarData(
            spots: spots,
            isCurved: true,
            color: Colors.blue,
            barWidth: 2,
            isStrokeCapRound: true,
            dotData: const FlDotData(show: false),
            belowBarData: BarAreaData(
              show: true,
              color: Colors.blue.withOpacity(0.1),
            ),
          ),
          // Threshold line
          LineChartBarData(
            spots: [
              FlSpot(spots.first.x, threshold),
              FlSpot(spots.last.x, threshold),
            ],
            isCurved: false,
            color: Colors.red,
            barWidth: 2,
            isStrokeCapRound: true,
            dotData: const FlDotData(show: false),
            dashArray: [5, 5],
          ),
        ],
        lineTouchData: LineTouchData(
          touchTooltipData: LineTouchTooltipData(
            getTooltipColor: (touchedSpot) => Colors.blueGrey.withOpacity(0.8),
            getTooltipItems: (List<LineBarSpot> touchedBarSpots) {
              return touchedBarSpots.map((barSpot) {
                final isThreshold = barSpot.barIndex == 1;
                if (isThreshold) {
                  return const LineTooltipItem(
                    'Threshold',
                    TextStyle(color: Colors.red),
                  );
                }
                return LineTooltipItem(
                  'Score: ${barSpot.y.toStringAsFixed(4)}\nTime: ${barSpot.x.toStringAsFixed(1)}s',
                  const TextStyle(color: Colors.white),
                );
              }).toList();
            },
          ),
        ),
      ),
    );
  }
}
