import 'dart:async';
import 'dart:math';
import 'package:flutter/material.dart';
import 'package:device_info_plus/device_info_plus.dart';
import 'package:mqtt_client/mqtt_client.dart';
import '../services/mqtt_service.dart';
import 'dart:convert';
class MqttSimulatorPage extends StatefulWidget {
  const MqttSimulatorPage({super.key});

  @override
  State<MqttSimulatorPage> createState() => _MqttSimulatorPageState();
}

class _MqttSimulatorPageState extends State<MqttSimulatorPage> {
  MQTTService? _mqttService;
  bool _isConnected = false;
  bool _isConnecting = false;
  bool _isPublishing = false;
  String _statusMessage = 'Disconnected';
  String _deviceId = '';
  String _hwId = '';
  Timer? _publishTimer;

  // Simulate ESP32 MSE pattern
  final List<double> _msePattern = [
    20, 22, 19, 23, 21, 24, 18, 25, 22, 20, 23, 19, 21, 24, 22,
    28, 32, 35, 36, // Anomaly spike
    25, 23, 21, 22, 20, 24, 19, 23, 22, 21
  ];
  int _patternIndex = 0;
  int _messageCount = 0;

  @override
  void initState() {
    super.initState();
    _initDeviceId();
  }

  @override
  void dispose() {
    _stopPublishing();
    _mqttService?.disconnect();
    super.dispose();
  }

  Future<void> _initDeviceId() async {
    final deviceInfo = DeviceInfoPlugin();
    final androidInfo = await deviceInfo.androidInfo;

    // Use last 4 chars of Android ID, similar to ESP32 MAC address
    final androidId = androidInfo.id;
    _hwId = androidId.substring(androidId.length - 4).toUpperCase();
    _deviceId = 'edgesonic-$_hwId'.toLowerCase();

    setState(() {
      _statusMessage = 'Device ID: $_deviceId (Code: $_hwId)';
    });
  }

  Future<void> _connectMqtt() async {
    setState(() {
      _isConnecting = true;
      _statusMessage = 'Connecting to MQTT...';
    });

    try {
      // Create MQTT service - connects to GCP through WireGuard
      _mqttService = MQTTService(
        broker: '10.8.0.1', // Through WireGuard VPN
        clientId: 'EdgeSonic-Android-$_hwId',
        port: 1883,
        keepAliveSeconds: 60,
      );

      await _mqttService!.connect();

      if (_mqttService!.client.connectionStatus!.state == MqttConnectionState.connected) {
        setState(() {
          _isConnected = true;
          _statusMessage = 'Connected to 10.8.0.1:1883';
        });

        // Publish metadata immediately after connecting
        await _publishMetadata();
      }
    } catch (e) {
      setState(() {
        _isConnected = false;
        _statusMessage = 'Connection failed: $e';
      });
    } finally {
      setState(() {
        _isConnecting = false;
      });
    }
  }

  void _disconnect() {
    _stopPublishing();
    _mqttService?.disconnect();
    setState(() {
      _isConnected = false;
      _statusMessage = 'Disconnected';
    });
  }

  Future<void> _publishMetadata() async {
    if (!_isConnected || _mqttService == null) return;

    final topic = 'sensors/$_deviceId/meta';
    final metadata = {
      'model': 'EdgeSonic Android Simulator',
      'fw': '1.0.0',
      'build': 'android',
      'hwId': _hwId,
      'deviceName': 'Android-Simulator',
      'friendlyName': 'Android-Test-$_hwId',
      'registrationCode': _hwId,
    };

    try {
      await _mqttService!.publishString(
        topic: topic,
        payload: jsonEncode(metadata),
      );
      debugPrint('Metadata published to $topic');
    } catch (e) {
      debugPrint('Failed to publish metadata: $e');
    }
  }

  Future<void> _publishTelemetry() async {
    if (!_isConnected || _mqttService == null) return;

    final topic = 'sensors/$_deviceId/telemetry';

    // Get MSE from pattern with slight random variation
    final baseMse = _msePattern[_patternIndex];
    final mse = (baseMse + Random().nextDouble() * 0.6 - 0.3).clamp(15.0, 50.0);
    _patternIndex = (_patternIndex + 1) % _msePattern.length;

    // Random SVDD score
    final svdd = Random().nextDouble() * 1.9 + 0.1;

    final telemetry = {
      'svdd': (svdd * 10000).round() / 10000.0,
      'mse': (mse * 100).round() / 100.0,
      'hwId': _hwId,
      'ts': DateTime.now().millisecondsSinceEpoch,
    };

    try {
      await _mqttService!.publishString(
        topic: topic,
        payload: jsonEncode(telemetry),
      );

      _messageCount++;
      final status = mse < 27 ? '[NORMAL]' : mse < 35 ? '[WARNING]' : '[ANOMALY]';

      setState(() {
        _statusMessage = '$status MSE: ${mse.toStringAsFixed(2)} | Messages: $_messageCount';
      });

      debugPrint('$status Published: MSE=${mse.toStringAsFixed(2)}, SVDD=${svdd.toStringAsFixed(4)}');
    } catch (e) {
      debugPrint('Failed to publish telemetry: $e');
    }
  }

  Future<void> _publishStatus() async {
    if (!_isConnected || _mqttService == null) return;

    final topic = 'sensors/$_deviceId/status';
    final status = {
      'uptime': DateTime.now().millisecondsSinceEpoch,
      'messages': _messageCount,
      'device': 'Android',
    };

    try {
      await _mqttService!.publishString(
        topic: topic,
        payload: jsonEncode(status),
      );
    } catch (e) {
      debugPrint('Failed to publish status: $e');
    }
  }

  void _startPublishing() {
    if (!_isConnected) return;

    setState(() {
      _isPublishing = true;
      _messageCount = 0;
    });

    // Publish telemetry every 3 seconds (like ESP32)
    _publishTimer = Timer.periodic(const Duration(seconds: 3), (timer) {
      _publishTelemetry();
    });
  }

  void _stopPublishing() {
    _publishTimer?.cancel();
    _publishTimer = null;
    setState(() {
      _isPublishing = false;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('MQTT ESP32 Simulator'),
        backgroundColor: Colors.teal,
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // Device Info Card
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16.0),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'Device Information',
                      style: Theme.of(context).textTheme.titleLarge,
                    ),
                    const SizedBox(height: 8),
                    Text('Device ID: $_deviceId'),
                    Text('Hardware ID: $_hwId'),
                    Text('Registration Code: $_hwId'),
                    const SizedBox(height: 8),
                    const Text(
                      '⚠️ Ensure WireGuard VPN is connected first!',
                      style: TextStyle(color: Colors.orange, fontWeight: FontWeight.bold),
                    ),
                  ],
                ),
              ),
            ),

            const SizedBox(height: 16),

            // Status Card
            Card(
              color: _isConnected ? Colors.green.shade50 : Colors.grey.shade100,
              child: Padding(
                padding: const EdgeInsets.all(16.0),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        Icon(
                          _isConnected ? Icons.check_circle : Icons.error,
                          color: _isConnected ? Colors.green : Colors.grey,
                        ),
                        const SizedBox(width: 8),
                        Text(
                          _isConnected ? 'Connected' : 'Disconnected',
                          style: Theme.of(context).textTheme.titleMedium,
                        ),
                      ],
                    ),
                    const SizedBox(height: 8),
                    Text(_statusMessage),
                    if (_isConnected) ...[
                      const SizedBox(height: 8),
                      Text('Messages sent: $_messageCount'),
                    ],
                  ],
                ),
              ),
            ),

            const SizedBox(height: 24),

            // Connection Controls
            if (!_isConnected) ...[
              ElevatedButton.icon(
                onPressed: _isConnecting ? null : _connectMqtt,
                icon: _isConnecting
                    ? const SizedBox(
                  width: 16,
                  height: 16,
                  child: CircularProgressIndicator(strokeWidth: 2),
                )
                    : const Icon(Icons.wifi),
                label: Text(_isConnecting ? 'Connecting...' : 'Connect to MQTT'),
                style: ElevatedButton.styleFrom(
                  padding: const EdgeInsets.all(16),
                  backgroundColor: Colors.teal,
                  foregroundColor: Colors.white,
                ),
              ),
            ] else ...[
              ElevatedButton.icon(
                onPressed: _disconnect,
                icon: const Icon(Icons.wifi_off),
                label: const Text('Disconnect'),
                style: ElevatedButton.styleFrom(
                  padding: const EdgeInsets.all(16),
                  backgroundColor: Colors.red,
                  foregroundColor: Colors.white,
                ),
              ),

              const SizedBox(height: 16),

              // Publishing Controls
              Row(
                children: [
                  Expanded(
                    child: ElevatedButton.icon(
                      onPressed: _isPublishing ? null : _startPublishing,
                      icon: const Icon(Icons.play_arrow),
                      label: const Text('Start Streaming'),
                      style: ElevatedButton.styleFrom(
                        padding: const EdgeInsets.all(16),
                      ),
                    ),
                  ),
                  const SizedBox(width: 8),
                  Expanded(
                    child: ElevatedButton.icon(
                      onPressed: _isPublishing ? _stopPublishing : null,
                      icon: const Icon(Icons.stop),
                      label: const Text('Stop Streaming'),
                      style: ElevatedButton.styleFrom(
                        padding: const EdgeInsets.all(16),
                      ),
                    ),
                  ),
                ],
              ),

              const SizedBox(height: 16),

              // Manual Publish
              OutlinedButton.icon(
                onPressed: _publishTelemetry,
                icon: const Icon(Icons.send),
                label: const Text('Send Single Message'),
                style: OutlinedButton.styleFrom(
                  padding: const EdgeInsets.all(16),
                ),
              ),
            ],

            const SizedBox(height: 24),

            // Instructions
            const Expanded(
              child: Card(
                child: Padding(
                  padding: EdgeInsets.all(16.0),
                  child: SingleChildScrollView(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          'Instructions:',
                          style: TextStyle(
                            fontSize: 16,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                        SizedBox(height: 8),
                        Text('1. Connect WireGuard VPN first (edgesonic_android)'),
                        Text('2. Click "Connect to MQTT"'),
                        Text('3. Click "Start Streaming" to simulate ESP32'),
                        Text('4. Watch data flow in your EdgeSonic dashboard'),
                        SizedBox(height: 16),
                        Text(
                          'Topics:',
                          style: TextStyle(fontWeight: FontWeight.bold),
                        ),
                        Text('• sensors/{deviceId}/meta - Device info'),
                        Text('• sensors/{deviceId}/telemetry - Sensor data'),
                        Text('• sensors/{deviceId}/status - Status updates'),
                      ],
                    ),
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}