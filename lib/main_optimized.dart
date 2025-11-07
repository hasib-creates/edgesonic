import 'dart:async';
import 'dart:math' as math;
import 'package:flutter/material.dart';
import 'package:file_picker/file_picker.dart';
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
      title: 'EdgeSonic',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: Colors.teal,
          brightness: Brightness.light,
        ),
        useMaterial3: true,
        cardTheme: CardTheme(
          elevation: 0,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(16),
            side: BorderSide(color: Colors.grey.shade200),
          ),
        ),
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

class _EdgeSonicHomePageState extends State<EdgeSonicHomePage>
    with SingleTickerProviderStateMixin {
  final AnomalyDetectionServiceOptimized _detector =
      AnomalyDetectionServiceOptimized();
  final LiveAudioServiceOptimized _liveAudioService =
      LiveAudioServiceOptimized();
  StreamSubscription<LiveAudioChunk>? _liveSubscription;

  late TabController _tabController;

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
    _tabController = TabController(length: 3, vsync: this);
    _loadModel();
  }

  @override
  void dispose() {
    _liveSubscription?.cancel();
    unawaited(_liveAudioService.stop());
    _liveAudioService.dispose();
    _detector.dispose();
    _tabController.dispose();
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
        _loadError = e.toString();
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
      setState(() => _liveError = 'Model not loaded');
      return;
    }

    setState(() => _liveError = null);

    final result = await _liveAudioService.start();
    if (!mounted) return;

    if (!result.started) {
      setState(() {
        _isLiveRunning = false;
        _liveError = result.error ?? 'Failed to start';
      });
      return;
    }

    _detector.clearBuffer();
    _detector.resetMetrics();

    await _liveSubscription?.cancel();
    _liveSubscription = _liveAudioService.chunks.listen(
      _handleLiveChunk,
      onError: (error) {
        if (!mounted) return;
        setState(() => _liveError = 'Error: $error');
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
    setState(() {
      _liveChunksProcessed = _liveAudioService.chunksProcessed;
      _liveRms = chunk.rms;
      _lastLiveChunkTime = chunk.timestamp;
    });

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
      setState(() => _liveError = 'Inference error: $e');
    } finally {
      _liveInferenceInProgress = false;
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('EdgeSonic'),
        centerTitle: true,
        elevation: 0,
        actions: [
          IconButton(
            icon: Icon(_modelLoaded ? Icons.check_circle : Icons.error_outline),
            color: _modelLoaded ? Colors.green : Colors.red,
            onPressed: _isLoadingModel ? null : _loadModel,
            tooltip: _modelLoaded ? 'Model Ready' : 'Tap to reload',
          ),
        ],
        bottom: TabBar(
          controller: _tabController,
          tabs: const [
            Tab(icon: Icon(Icons.mic), text: 'Live'),
            Tab(icon: Icon(Icons.upload_file), text: 'File'),
            Tab(icon: Icon(Icons.router), text: 'MQTT'),
          ],
        ),
      ),
      body: TabBarView(
        controller: _tabController,
        children: [
          _buildLiveTab(),
          _buildFileTab(),
          _buildMqttTab(),
        ],
      ),
    );
  }

  // ========== LIVE TAB ==========
  Widget _buildLiveTab() {
    final theme = Theme.of(context);
    final result = _latestLiveResult;
    final isAnomaly = result?.isAnomaly ?? false;

    return SingleChildScrollView(
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          // Status Card
          Card(
            color: _isLiveRunning
                ? (isAnomaly ? Colors.red.shade50 : Colors.green.shade50)
                : null,
            child: Padding(
              padding: const EdgeInsets.all(20),
              child: Column(
                children: [
                  Icon(
                    _isLiveRunning ? Icons.mic : Icons.mic_off,
                    size: 64,
                    color: _isLiveRunning
                        ? (isAnomaly ? Colors.red : Colors.green)
                        : Colors.grey,
                  ),
                  const SizedBox(height: 12),
                  Text(
                    _isLiveRunning ? 'LISTENING' : 'STOPPED',
                    style: theme.textTheme.headlineSmall?.copyWith(
                      fontWeight: FontWeight.bold,
                      color: _isLiveRunning
                          ? (isAnomaly ? Colors.red : Colors.green)
                          : Colors.grey,
                    ),
                  ),
                  if (result != null && _isLiveRunning) ...[
                    const SizedBox(height: 8),
                    Text(
                      isAnomaly ? '⚠️ ANOMALY DETECTED' : '✓ Normal',
                      style: theme.textTheme.titleMedium?.copyWith(
                        color: isAnomaly ? Colors.red : Colors.green,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                  ],
                ],
              ),
            ),
          ),

          const SizedBox(height: 16),

          // Metrics Grid
          if (_isLiveRunning) ...[
            Row(
              children: [
                Expanded(
                  child: _buildMetricCard(
                    'Chunks',
                    '$_liveChunksProcessed',
                    Icons.analytics,
                    Colors.blue,
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: _buildMetricCard(
                    'RMS',
                    _liveRms?.toStringAsFixed(3) ?? '---',
                    Icons.graphic_eq,
                    Colors.purple,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 12),
            if (result != null) ...[
              Row(
                children: [
                  Expanded(
                    child: _buildMetricCard(
                      'Score',
                      result.smoothedScore.toStringAsFixed(4),
                      Icons.score,
                      isAnomaly ? Colors.red : Colors.green,
                    ),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: _buildMetricCard(
                      'Latency',
                      '${result.processingTimeMs.toStringAsFixed(1)}ms',
                      Icons.speed,
                      Colors.orange,
                    ),
                  ),
                ],
              ),
            ],
            const SizedBox(height: 20),
          ],

          // Control Button
          FilledButton.icon(
            onPressed: _modelLoaded ? _toggleLiveCapture : null,
            icon: Icon(_isLiveRunning ? Icons.stop : Icons.play_arrow),
            label: Text(
              _isLiveRunning ? 'Stop Capture' : 'Start Live Capture',
              style: const TextStyle(fontSize: 16),
            ),
            style: FilledButton.styleFrom(
              padding: const EdgeInsets.symmetric(vertical: 16),
              backgroundColor: _isLiveRunning ? Colors.red : null,
            ),
          ),

          // Error Display
          if (_liveError != null) ...[
            const SizedBox(height: 16),
            Card(
              color: Colors.red.shade50,
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Row(
                  children: [
                    Icon(Icons.error, color: Colors.red.shade700),
                    const SizedBox(width: 12),
                    Expanded(
                      child: Text(
                        _liveError!,
                        style: TextStyle(color: Colors.red.shade700),
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ],

          // Info Card
          if (!_isLiveRunning && _liveError == null) ...[
            const SizedBox(height: 16),
            Card(
              color: Colors.blue.shade50,
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        Icon(Icons.info_outline, color: Colors.blue.shade700),
                        const SizedBox(width: 8),
                        Text(
                          'How it works',
                          style: TextStyle(
                            color: Colors.blue.shade700,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 8),
                    Text(
                      '• Captures audio at 16kHz\n'
                      '• Analyzes 128-frame windows\n'
                      '• Detects anomalies in real-time\n'
                      '• 5-15ms latency',
                      style: TextStyle(
                        color: Colors.blue.shade700,
                        fontSize: 13,
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ],
        ],
      ),
    );
  }

  Widget _buildMetricCard(String label, String value, IconData icon, Color color) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          children: [
            Icon(icon, color: color, size: 28),
            const SizedBox(height: 8),
            Text(
              value,
              style: TextStyle(
                fontSize: 20,
                fontWeight: FontWeight.bold,
                color: color,
              ),
            ),
            const SizedBox(height: 4),
            Text(
              label,
              style: TextStyle(
                fontSize: 12,
                color: Colors.grey.shade600,
              ),
            ),
          ],
        ),
      ),
    );
  }

  // ========== FILE TAB ==========
  Widget _buildFileTab() {
    final theme = Theme.of(context);
    return SingleChildScrollView(
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          // File Selection Card
          Card(
            child: Padding(
              padding: const EdgeInsets.all(20),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Icon(Icons.audio_file, color: theme.colorScheme.primary),
                      const SizedBox(width: 8),
                      Text(
                        'Audio File Processing',
                        style: theme.textTheme.titleLarge,
                      ),
                    ],
                  ),
                  const SizedBox(height: 16),
                  if (_selectedFileName != null) ...[
                    Container(
                      padding: const EdgeInsets.all(12),
                      decoration: BoxDecoration(
                        color: Colors.grey.shade100,
                        borderRadius: BorderRadius.circular(8),
                      ),
                      child: Row(
                        children: [
                          const Icon(Icons.audio_file, size: 20),
                          const SizedBox(width: 8),
                          Expanded(
                            child: Text(
                              _selectedFileName!,
                              style: const TextStyle(fontWeight: FontWeight.w500),
                            ),
                          ),
                        ],
                      ),
                    ),
                    const SizedBox(height: 16),
                  ],
                  FilledButton.icon(
                    onPressed: _isProcessingFile ? null : _pickAudioFile,
                    icon: const Icon(Icons.folder_open),
                    label: const Text('Select Audio File'),
                    style: FilledButton.styleFrom(
                      padding: const EdgeInsets.symmetric(vertical: 14),
                    ),
                  ),
                  if (_selectedFilePath != null) ...[
                    const SizedBox(height: 12),
                    OutlinedButton.icon(
                      onPressed: null, // TODO: Implement file processing
                      icon: const Icon(Icons.play_arrow),
                      label: const Text('Process (Coming Soon)'),
                      style: OutlinedButton.styleFrom(
                        padding: const EdgeInsets.symmetric(vertical: 14),
                      ),
                    ),
                  ],
                ],
              ),
            ),
          ),

          if (_isProcessingFile) ...[
            const SizedBox(height: 16),
            Card(
              child: Padding(
                padding: const EdgeInsets.all(20),
                child: Column(
                  children: [
                    CircularProgressIndicator(value: _fileProgress),
                    const SizedBox(height: 12),
                    Text('Processing: ${(_fileProgress * 100).toStringAsFixed(0)}%'),
                  ],
                ),
              ),
            ),
          ],

          if (_fileError != null) ...[
            const SizedBox(height: 16),
            Card(
              color: Colors.orange.shade50,
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Row(
                  children: [
                    Icon(Icons.info, color: Colors.orange.shade700),
                    const SizedBox(width: 12),
                    Expanded(
                      child: Text(
                        _fileError!,
                        style: TextStyle(color: Colors.orange.shade700),
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ],

          // Info Card
          const SizedBox(height: 16),
          Card(
            color: Colors.blue.shade50,
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Icon(Icons.info_outline, color: Colors.blue.shade700),
                      const SizedBox(width: 8),
                      Text(
                        'File Processing',
                        style: TextStyle(
                          color: Colors.blue.shade700,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 8),
                  Text(
                    'Upload audio files to analyze for anomalies offline. '
                    'Results will be displayed with timestamps and exportable to CSV.',
                    style: TextStyle(
                      color: Colors.blue.shade700,
                      fontSize: 13,
                    ),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  // ========== MQTT TAB ==========
  Widget _buildMqttTab() {
    final theme = Theme.of(context);
    return SingleChildScrollView(
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Card(
            child: Padding(
              padding: const EdgeInsets.all(20),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Icon(Icons.router, color: theme.colorScheme.primary, size: 32),
                      const SizedBox(width: 12),
                      Text(
                        'MQTT Integration',
                        style: theme.textTheme.titleLarge,
                      ),
                    ],
                  ),
                  const SizedBox(height: 16),
                  Text(
                    'Connect to your MQTT broker to receive telemetry '
                    'or simulate ESP32 device payloads.',
                    style: theme.textTheme.bodyMedium,
                  ),
                  const SizedBox(height: 20),
                  FilledButton.icon(
                    onPressed: () {
                      Navigator.push(
                        context,
                        MaterialPageRoute(builder: (_) => const MqttTestPage()),
                      );
                    },
                    icon: const Icon(Icons.wifi_tethering),
                    label: const Text('MQTT Connection Test'),
                    style: FilledButton.styleFrom(
                      padding: const EdgeInsets.symmetric(vertical: 14),
                    ),
                  ),
                  const SizedBox(height: 12),
                  OutlinedButton.icon(
                    onPressed: () {
                      Navigator.push(
                        context,
                        MaterialPageRoute(builder: (_) => const MqttSimulatorPage()),
                      );
                    },
                    icon: const Icon(Icons.sensors),
                    label: const Text('ESP32 Simulator'),
                    style: OutlinedButton.styleFrom(
                      padding: const EdgeInsets.symmetric(vertical: 14),
                    ),
                  ),
                ],
              ),
            ),
          ),

          const SizedBox(height: 16),

          // Features Card
          Card(
            color: Colors.green.shade50,
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Icon(Icons.check_circle, color: Colors.green.shade700),
                      const SizedBox(width: 8),
                      Text(
                        'Features',
                        style: TextStyle(
                          color: Colors.green.shade700,
                          fontWeight: FontWeight.bold,
                          fontSize: 16,
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 12),
                  _buildFeatureItem('Connect to any MQTT broker'),
                  _buildFeatureItem('Subscribe to topics'),
                  _buildFeatureItem('Publish anomaly results'),
                  _buildFeatureItem('Simulate ESP32 telemetry'),
                  _buildFeatureItem('Real-time message monitoring'),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildFeatureItem(String text) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        children: [
          Icon(Icons.check, color: Colors.green.shade700, size: 18),
          const SizedBox(width: 8),
          Expanded(
            child: Text(
              text,
              style: TextStyle(color: Colors.green.shade700),
            ),
          ),
        ],
      ),
    );
  }
}
