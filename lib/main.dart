import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:file_picker/file_picker.dart';
import 'dart:typed_data';
import 'package:path_provider/path_provider.dart';

import 'optimized/anomaly_detection_service_optimized.dart';
import 'optimized/audio_processing_service_optimized.dart';
import 'services/live_audio_service_optimized.dart';
import 'services/mqtt_service.dart';

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

class _EdgeSonicHomePageState extends State<EdgeSonicHomePage>
    with SingleTickerProviderStateMixin {
  final AnomalyDetectionServiceOptimized _detector =
      AnomalyDetectionServiceOptimized();
  final LiveAudioServiceOptimized _liveAudioService =
      LiveAudioServiceOptimized();
  StreamSubscription<LiveAudioChunk>? _liveSubscription;

  late TabController _tabController;

  bool _modelLoaded = false;
  String? _modelName;
  String? _modelPath;

  // Live state
  bool _isLiveRunning = false;
  bool _liveInferenceInProgress = false;
  int _liveChunksProcessed = 0;
  AnomalyResult? _latestLiveResult;
  String? _liveError;

  // File state
  String? _selectedFileName;
  bool _isProcessingFile = false;
  List<AnomalyResult>? _fileResults;
  String? _fileError;

  // MQTT state
  final _brokerController = TextEditingController(text: '10.8.0.1');
  final _hwIdController = TextEditingController(text: 'ABCD');
  bool _isSendingMqtt = false;
  String? _mqttStatus;

  @override
  void initState() {
    super.initState();
    _tabController = TabController(length: 3, vsync: this);
  }

  @override
  void dispose() {
    _liveSubscription?.cancel();
    unawaited(_liveAudioService.stop());
    _liveAudioService.dispose();
    _detector.dispose();
    _tabController.dispose();
    _brokerController.dispose();
    _hwIdController.dispose();
    super.dispose();
  }

  Future<void> _pickAndLoadModel() async {
    final result = await FilePicker.platform.pickFiles(
      type: FileType.custom,
      allowedExtensions: ['tflite'],
    );

    if (result?.files.single.path == null) return;

    try {
      final sourcePath = result!.files.single.path!;

      // Copy model to app's document directory for persistent access
      final appDir = await getApplicationDocumentsDirectory();
      final targetPath = '${appDir.path}/model.tflite';

      final sourceFile = File(sourcePath);
      await sourceFile.copy(targetPath);

      // Load the model
      final loaded = await _detector.loadModel(modelPath: targetPath);

      if (!mounted) return;
      setState(() {
        _modelLoaded = loaded;
        _modelName = result.files.single.name;
        _modelPath = targetPath;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _modelLoaded = false;
        _modelName = null;
      });

      // Show error dialog
      if (mounted) {
        showDialog(
          context: context,
          builder: (context) => AlertDialog(
            title: const Text('Error'),
            content: Text('Failed to load model: $e'),
            actions: [
              TextButton(
                onPressed: () => Navigator.pop(context),
                child: const Text('OK'),
              ),
            ],
          ),
        );
      }
    }
  }

  Future<void> _pickAndProcessAudioFile() async {
    final result = await FilePicker.platform.pickFiles(
      type: FileType.custom,
      allowedExtensions: ['wav', 'mp3', 'm4a', 'flac'],
    );

    if (result?.files.single.path == null) return;

    setState(() {
      _selectedFileName = result!.files.single.name;
      _isProcessingFile = true;
      _fileResults = null;
      _fileError = null;
    });

    try {
      final file = File(result!.files.single.path!);
      final bytes = await file.readAsBytes();

      // Process audio
      final results = await _processAudioBytes(bytes);

      if (!mounted) return;
      setState(() {
        _fileResults = results;
        _isProcessingFile = false;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _fileError = e.toString();
        _isProcessingFile = false;
      });
    }
  }

  Future<List<AnomalyResult>> _processAudioBytes(Uint8List bytes) async {
    // Skip WAV header (44 bytes)
    final audioData = bytes.sublist(44);

    // Convert to Float32
    final samples = Float32List(audioData.length ~/ 2);
    for (int i = 0; i < samples.length; i++) {
      int low = audioData[i * 2];
      int high = audioData[i * 2 + 1] << 8;
      int sample = high | low;
      if (sample & 0x8000 != 0) sample = sample - 0x10000;
      samples[i] = sample / 32768.0;
    }

    // Process in chunks
    final results = <AnomalyResult>[];
    final chunkSize = AudioProcessingServiceOptimized.samplesPerChunk;
    final hopSize = AudioProcessingServiceOptimized.hopSamples;

    for (int i = 0; i + chunkSize < samples.length; i += hopSize) {
      final chunk = Float32List.sublistView(samples, i, i + chunkSize);
      final result = await _detector.processAudioChunk(chunk);
      results.add(result);
    }

    return results;
  }

  Future<void> _toggleLiveCapture() async {
    if (_isLiveRunning) {
      await _stopLiveCapture();
    } else {
      await _startLiveCapture();
    }
  }

  Future<void> _startLiveCapture() async {
    if (!_modelLoaded) return;

    setState(() => _liveError = null);

    final result = await _liveAudioService.start();
    if (!mounted) return;

    if (!result.started) {
      setState(() {
        _isLiveRunning = false;
        _liveError = result.error;
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
        setState(() => _liveError = error.toString());
        unawaited(_stopLiveCapture());
      },
    );

    setState(() {
      _isLiveRunning = true;
      _liveChunksProcessed = 0;
      _latestLiveResult = null;
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
    setState(() => _liveChunksProcessed = _liveAudioService.chunksProcessed);

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
      setState(() => _liveError = e.toString());
    } finally {
      _liveInferenceInProgress = false;
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(_modelLoaded ? _modelName ?? 'EdgeSonic' : 'EdgeSonic'),
        centerTitle: true,
        actions: [
          IconButton(
            icon: Icon(_modelLoaded ? Icons.check_circle : Icons.upload),
            color: _modelLoaded ? Colors.green : Colors.grey,
            onPressed: _pickAndLoadModel,
            tooltip: 'Upload Model',
          ),
        ],
        bottom: _modelLoaded
            ? TabBar(
                controller: _tabController,
                tabs: const [
                  Tab(icon: Icon(Icons.mic), text: 'Live'),
                  Tab(icon: Icon(Icons.upload_file), text: 'File'),
                  Tab(icon: Icon(Icons.cloud_upload), text: 'MQTT'),
                ],
              )
            : null,
      ),
      body: _modelLoaded
          ? TabBarView(
              controller: _tabController,
              children: [
                _buildLiveTab(),
                _buildFileTab(),
                _buildMqttTab(),
              ],
            )
          : _buildUploadPrompt(),
    );
  }

  Widget _buildUploadPrompt() {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(Icons.upload_file, size: 80, color: Colors.grey.shade400),
          const SizedBox(height: 24),
          const Text(
            'Upload TFLite Model',
            style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
          ),
          const SizedBox(height: 32),
          FilledButton.icon(
            onPressed: _pickAndLoadModel,
            icon: const Icon(Icons.upload),
            label: const Text('Select Model'),
            style: FilledButton.styleFrom(minimumSize: const Size(200, 48)),
          ),
        ],
      ),
    );
  }

  Widget _buildLiveTab() {
    final result = _latestLiveResult;
    final isAnomaly = result?.isAnomaly ?? false;

    return Padding(
      padding: const EdgeInsets.all(16),
      child: Column(
        children: [
          Icon(
            _isLiveRunning ? Icons.mic : Icons.mic_off,
            size: 80,
            color: _isLiveRunning
                ? (isAnomaly ? Colors.red : Colors.green)
                : Colors.grey,
          ),
          const SizedBox(height: 8),
          Text(
            _isLiveRunning ? (isAnomaly ? 'ANOMALY' : 'NORMAL') : 'STOPPED',
            style: TextStyle(
              fontSize: 24,
              fontWeight: FontWeight.bold,
              color: _isLiveRunning
                  ? (isAnomaly ? Colors.red : Colors.green)
                  : Colors.grey,
            ),
          ),
          const SizedBox(height: 24),
          if (_isLiveRunning && result != null) ...[
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: [
                _buildMetric('Chunks', '$_liveChunksProcessed'),
                _buildMetric('Score', result.smoothedScore.toStringAsFixed(4)),
                _buildMetric('Latency',
                    '${result.processingTimeMs.toStringAsFixed(0)}ms'),
              ],
            ),
            const SizedBox(height: 24),
          ],
          FilledButton.icon(
            onPressed: _toggleLiveCapture,
            icon: Icon(_isLiveRunning ? Icons.stop : Icons.play_arrow),
            label: Text(_isLiveRunning ? 'Stop' : 'Start'),
            style: FilledButton.styleFrom(
              minimumSize: const Size(200, 48),
              backgroundColor: _isLiveRunning ? Colors.red : null,
            ),
          ),
          if (_liveError != null) ...[
            const SizedBox(height: 16),
            Text(_liveError!, style: const TextStyle(color: Colors.red)),
          ],
        ],
      ),
    );
  }

  Widget _buildMetric(String label, String value) {
    return Column(
      children: [
        Text(value,
            style: const TextStyle(fontSize: 20, fontWeight: FontWeight.bold)),
        Text(label,
            style: TextStyle(fontSize: 12, color: Colors.grey.shade600)),
      ],
    );
  }

  Widget _buildFileTab() {
    return Padding(
      padding: const EdgeInsets.all(16),
      child: Column(
        children: [
          const SizedBox(height: 20),
          if (_selectedFileName != null) ...[
            Text(_selectedFileName!,
                style: const TextStyle(fontWeight: FontWeight.w500)),
            const SizedBox(height: 16),
          ],
          FilledButton.icon(
            onPressed: _isProcessingFile ? null : _pickAndProcessAudioFile,
            icon: const Icon(Icons.upload_file),
            label: const Text('Select & Process'),
            style: FilledButton.styleFrom(minimumSize: const Size(200, 48)),
          ),
          const SizedBox(height: 24),
          if (_isProcessingFile) const CircularProgressIndicator(),
          if (_fileResults != null) ...[
            Text(
              '${_fileResults!.length} chunks',
              style: const TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 8),
            Text(
              '${_fileResults!.where((r) => r.isAnomaly).length} anomalies',
              style: const TextStyle(color: Colors.red, fontSize: 16),
            ),
          ],
          if (_fileError != null)
            Text(_fileError!, style: const TextStyle(color: Colors.red)),
        ],
      ),
    );
  }

  Widget _buildMqttTab() {
    return Padding(
      padding: const EdgeInsets.all(16),
      child: Column(
        children: [
          const SizedBox(height: 20),
          TextField(
            controller: _brokerController,
            decoration: const InputDecoration(
              labelText: 'Broker',
              border: OutlineInputBorder(),
            ),
          ),
          const SizedBox(height: 12),
          TextField(
            controller: _hwIdController,
            decoration: const InputDecoration(
              labelText: 'Hardware ID',
              border: OutlineInputBorder(),
            ),
          ),
          const SizedBox(height: 24),
          FilledButton.icon(
            onPressed: _isSendingMqtt ? null : _sendMqttData,
            icon: const Icon(Icons.send),
            label: Text(_isSendingMqtt ? 'Sending...' : 'Send'),
            style: FilledButton.styleFrom(minimumSize: const Size(200, 48)),
          ),
          const SizedBox(height: 16),
          if (_mqttStatus != null)
            Text(
              _mqttStatus!,
              style: TextStyle(
                color: _mqttStatus!.contains('Error') ? Colors.red : Colors.green,
              ),
            ),
        ],
      ),
    );
  }

  Future<void> _sendMqttData() async {
    final broker = _brokerController.text.trim();
    final hwId = _hwIdController.text.trim();

    if (broker.isEmpty || hwId.isEmpty) {
      setState(() => _mqttStatus = 'Error: Broker and HW ID required');
      return;
    }

    setState(() {
      _isSendingMqtt = true;
      _mqttStatus = null;
    });

    final deviceId = 'edgesonic-${hwId.toLowerCase()}';
    final mqtt = MQTTService(
      broker: broker,
      clientId: 'EdgeSonic-$hwId',
    );

    try {
      await mqtt.connect();

      final result = _latestLiveResult;
      final now = DateTime.now().millisecondsSinceEpoch;

      final telemetryPayload = {
        'mse': result?.smoothedScore.toStringAsFixed(4) ?? '0.0',
        'anomaly': result?.isAnomaly ?? false,
        'hwId': hwId,
        'ts': now,
      };

      await mqtt.publishString(
        topic: 'sensors/$deviceId/telemetry',
        payload: jsonEncode(telemetryPayload),
      );

      if (!mounted) return;
      setState(() => _mqttStatus = 'Sent successfully');
    } catch (e) {
      if (!mounted) return;
      setState(() => _mqttStatus = 'Error: $e');
    } finally {
      mqtt.disconnect();
      if (mounted) {
        setState(() => _isSendingMqtt = false);
      }
    }
  }
}
