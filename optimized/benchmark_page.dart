import 'dart:async';
import 'dart:typed_data';
import 'dart:math' as math;
import 'package:flutter/material.dart';
import 'anomaly_detection_service_optimized.dart';
import 'audio_processing_service_optimized.dart';

/// Comprehensive benchmarking page for real-time anomaly detection
class BenchmarkPage extends StatefulWidget {
  const BenchmarkPage({super.key});

  @override
  State<BenchmarkPage> createState() => _BenchmarkPageState();
}

class _BenchmarkPageState extends State<BenchmarkPage> {
  late AnomalyDetectionServiceOptimized _detector;
  bool _isLoaded = false;
  bool _isRunning = false;
  
  // Benchmark results
  final List<double> _inferenceT imes = [];
  final List<double> _audioProcessingTimes = [];
  final List<double> _totalTimes = [];
  
  String _status = 'Not started';
  int _chunksProcessed = 0;
  
  @override
  void initState() {
    super.initState();
    _loadModel();
  }
  
  Future<void> _loadModel() async {
    setState(() => _status = 'Loading model...');
    
    _detector = AnomalyDetectionServiceOptimized();
    final loaded = await _detector.loadModel();
    
    setState(() {
      _isLoaded = loaded;
      _status = loaded ? 'Model loaded' : 'Failed to load model';
    });
  }
  
  /// Run comprehensive benchmark
  Future<void> _runBenchmark() async {
    if (!_isLoaded || _isRunning) return;
    
    setState(() {
      _isRunning = true;
      _status = 'Running benchmark...';
      _chunksProcessed = 0;
      _inferenceTime s.clear();
      _audioProcessingTimes.clear();
      _totalTimes.clear();
    });
    
    _detector.resetMetrics();
    
    // Generate test audio chunks (simulate real-time streaming)
    const int numChunks = 100;
    final chunkSize = AudioProcessingServiceOptimized.samplesPerChunk;
    
    for (int i = 0; i < numChunks && _isRunning; i++) {
      final totalStopwatch = Stopwatch()..start();
      
      // Generate synthetic audio (mix of sine waves + noise)
      final audioChunk = _generateTestAudio(chunkSize, seed: i);
      
      // Process chunk
      final result = await _detector.processAudioChunk(audioChunk);
      
      totalStopwatch.stop();
      
      _totalTimes.add(totalStopwatch.elapsedMicroseconds / 1000.0);
      
      setState(() {
        _chunksProcessed = i + 1;
        _status = 'Processed ${ _chunksProcessed}/$numChunks chunks\n'
                  'Latest score: ${result.smoothedScore.toStringAsFixed(4)}';
      });
      
      // Simulate real-time processing delay
      await Future.delayed(const Duration(milliseconds: 10));
    }
    
    // Get final metrics
    final metrics = _detector.getPerformanceMetrics();
    final audioMetrics = metrics['audio_processing'] as Map<String, dynamic>;
    
    setState(() {
      _isRunning = false;
      _status = _buildSummary(metrics, audioMetrics);
    });
  }
  
  String _buildSummary(Map<String, dynamic> metrics, Map<String, dynamic> audioMetrics) {
    final avgTotal = _totalTimes.isEmpty ? 0.0 : _totalTimes.reduce((a, b) => a + b) / _totalTimes.length;
    final minTotal = _totalTimes.isEmpty ? 0.0 : _totalTimes.reduce(math.min);
    final maxTotal = _totalTimes.isEmpty ? 0.0 : _totalTimes.reduce(math.max);
    
    final avgInference = metrics['avg_inference_ms'] ?? 0.0;
    final avgAudioProcessing = audioMetrics['avg_time_ms'] ?? 0.0;
    
    // Calculate real-time factor
    final chunkDurationMs = (AudioProcessingServiceOptimized.samplesPerChunk / AudioConfig.sampleRate) * 1000;
    final realtimeFactor = avgTotal > 0 ? chunkDurationMs / avgTotal : 0.0;
    
    return '''
BENCHMARK RESULTS (${ _chunksProcessed} chunks)
────────────────────────────────────
Total Processing Time:
  Average: ${avgTotal.toStringAsFixed(2)} ms
  Min: ${minTotal.toStringAsFixed(2)} ms
  Max: ${maxTotal.toStringAsFixed(2)} ms

Inference Time:
  Average: ${avgInference.toStringAsFixed(2)} ms
  FPS: ${metrics['inference_fps'].toStringAsFixed(1)}

Audio Processing:
  Average: ${avgAudioProcessing.toStringAsFixed(2)} ms
  FPS: ${audioMetrics['fps'].toStringAsFixed(1)}

Real-Time Performance:
  Chunk duration: ${chunkDurationMs.toStringAsFixed(2)} ms
  Real-time factor: ${realtimeFactor.toStringAsFixed(2)}x
  ${realtimeFactor >= 1.0 ? '✅ CAN RUN IN REAL-TIME' : '❌ TOO SLOW FOR REAL-TIME'}

Latency Budget:
  Processing: ${avgTotal.toStringAsFixed(2)} ms
  Available: ${chunkDurationMs.toStringAsFixed(2)} ms
  Overhead: ${(chunkDurationMs - avgTotal).toStringAsFixed(2)} ms
''';
  }
  
  Float32List _generateTestAudio(int length, {int seed = 0}) {
    final random = math.Random(seed);
    final audio = Float32List(length);
    
    // Mix of sine waves at different frequencies
    for (int i = 0; i < length; i++) {
      final t = i / AudioConfig.sampleRate;
      audio[i] = (
        0.3 * math.sin(2 * math.pi * 440 * t) +  // A4
        0.2 * math.sin(2 * math.pi * 880 * t) +  // A5
        0.1 * (random.nextDouble() * 2 - 1)       // Noise
      );
    }
    
    return audio;
  }
  
  void _stopBenchmark() {
    setState(() {
      _isRunning = false;
      _status = 'Benchmark stopped';
    });
  }
  
  @override
  void dispose() {
    _detector.dispose();
    super.dispose();
  }
  
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Real-Time Performance Benchmark'),
        backgroundColor: Colors.deepPurple,
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // Status Card
            Card(
              color: _isLoaded ? Colors.green.shade50 : Colors.grey.shade100,
              child: Padding(
                padding: const EdgeInsets.all(16.0),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        Icon(
                          _isLoaded ? Icons.check_circle : Icons.error,
                          color: _isLoaded ? Colors.green : Colors.red,
                        ),
                        const SizedBox(width: 8),
                        Text(
                          _isLoaded ? 'Model Ready' : 'Model Not Loaded',
                          style: Theme.of(context).textTheme.titleMedium,
                        ),
                      ],
                    ),
                    if (_isLoaded) ...[
                      const SizedBox(height: 8),
                      Text('Threshold: ${_detector.threshold.toStringAsFixed(4)}'),
                      Text('Chunk size: ${AudioProcessingServiceOptimized.samplesPerChunk} samples'),
                      Text('Chunk duration: ${(AudioProcessingServiceOptimized.samplesPerChunk / AudioConfig.sampleRate * 1000).toStringAsFixed(1)} ms'),
                    ],
                  ],
                ),
              ),
            ),
            
            const SizedBox(height: 16),
            
            // Control Buttons
            Row(
              children: [
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: _isLoaded && !_isRunning ? _runBenchmark : null,
                    icon: const Icon(Icons.play_arrow),
                    label: const Text('Run Benchmark'),
                    style: ElevatedButton.styleFrom(
                      padding: const EdgeInsets.all(16),
                      backgroundColor: Colors.deepPurple,
                      foregroundColor: Colors.white,
                    ),
                  ),
                ),
                const SizedBox(width: 8),
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: _isRunning ? _stopBenchmark : null,
                    icon: const Icon(Icons.stop),
                    label: const Text('Stop'),
                    style: ElevatedButton.styleFrom(
                      padding: const EdgeInsets.all(16),
                      backgroundColor: Colors.red,
                      foregroundColor: Colors.white,
                    ),
                  ),
                ),
              ],
            ),
            
            const SizedBox(height: 16),
            
            // Results Card
            Expanded(
              child: Card(
                child: Padding(
                  padding: const EdgeInsets.all(16.0),
                  child: SingleChildScrollView(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          'Benchmark Results',
                          style: Theme.of(context).textTheme.titleLarge,
                        ),
                        const SizedBox(height: 16),
                        if (_isRunning)
                          const Center(child: CircularProgressIndicator())
                        else
                          Text(
                            _status,
                            style: const TextStyle(
                              fontFamily: 'monospace',
                              fontSize: 12,
                            ),
                          ),
                      ],
                    ),
                  ),
                ),
              ),
            ),
            
            const SizedBox(height: 16),
            
            // Instructions
            const Card(
              color: Colors.blue50,
              child: Padding(
                padding: EdgeInsets.all(12.0),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'Real-Time Requirements:',
                      style: TextStyle(fontWeight: FontWeight.bold),
                    ),
                    SizedBox(height: 4),
                    Text('• Processing time must be < chunk duration', style: TextStyle(fontSize: 12)),
                    Text('• Real-time factor should be > 1.0x', style: TextStyle(fontSize: 12)),
                    Text('• Target: <50ms latency for responsive system', style: TextStyle(fontSize: 12)),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
