import 'dart:async';
import 'dart:convert';

import 'package:flutter/material.dart';

import '../services/mqtt_service.dart';

class MqttTestPage extends StatefulWidget {
  const MqttTestPage({super.key});

  @override
  State<MqttTestPage> createState() => _MqttTestPageState();
}

class _MqttTestPageState extends State<MqttTestPage> {
  final _brokerController = TextEditingController(text: '10.8.0.1');
  final _portController = TextEditingController(text: '1883');
  final _deviceNameController = TextEditingController(text: 'EdgeSonic-1');
  final _friendlyNameController = TextEditingController(text: 'Aircon-1');
  final _hwIdController = TextEditingController(text: 'ABCD');
  final _registrationCodeController = TextEditingController(text: 'ABCD');
  final _logs = <String>[];

  bool _isSending = false;

  @override
  void dispose() {
    _brokerController.dispose();
    _portController.dispose();
    _deviceNameController.dispose();
    _friendlyNameController.dispose();
    _hwIdController.dispose();
    _registrationCodeController.dispose();
    super.dispose();
  }

  String get _deviceId => 'edgesonic-${_hwIdController.text.trim().toLowerCase()}';

  void _appendLog(String message) {
    setState(() {
      _logs.insert(0, '${DateTime.now().toIso8601String()}  $message');
    });
  }

  Map<String, dynamic> _buildMetaPayload() {
    return {
      'model': 'EdgeSonic Sensor v1.0',
      'fw': '1.0.0',
      'build': 'production',
      'hwId': _hwIdController.text.trim(),
      'deviceName': _deviceNameController.text.trim(),
      'friendlyName': _friendlyNameController.text.trim(),
      'registrationCode': _registrationCodeController.text.trim(),
    };
  }

  Map<String, dynamic> _buildStatusPayload() {
    final now = DateTime.now().millisecondsSinceEpoch;
    return {
      'uptime': now,
      'rssi': -42,
      'wifi': 'wireguard',
      'freeHeap': 128000,
    };
  }

  Map<String, dynamic> _buildTelemetryPayload() {
    final now = DateTime.now().millisecondsSinceEpoch;
    const mse = 29.6;
    const svdd = 0.73;
    return {
      'svdd': double.parse(svdd.toStringAsFixed(4)),
      'mse': double.parse(mse.toStringAsFixed(2)),
      'hwId': _hwIdController.text.trim(),
      'ts': now,
    };
  }

  Future<void> _sendFakeData() async {
    final broker = _brokerController.text.trim();
    final port = int.tryParse(_portController.text.trim()) ?? 1883;
    final hwId = _hwIdController.text.trim();
    if (broker.isEmpty || hwId.isEmpty) {
      _appendLog('Broker address and hardware ID are required.');
      return;
    }

    setState(() {
      _isSending = true;
    });

    final clientId = '${_deviceNameController.text.trim()}-$hwId';
    final mqtt = MQTTService(
      broker: broker,
      port: port,
      clientId: clientId,
    );

    try {
      _appendLog('Connecting to $broker:$port …');
      await mqtt.connect();
      _appendLog('Connected as $clientId');

      final metaTopic = 'sensors/$_deviceId/meta';
      final statusTopic = 'sensors/$_deviceId/status';
      final telemetryTopic = 'sensors/$_deviceId/telemetry';

      await mqtt.publishString(
        topic: metaTopic,
        payload: jsonEncode(_buildMetaPayload()),
      );
      _appendLog('Meta published to $metaTopic');

      await mqtt.publishString(
        topic: statusTopic,
        payload: jsonEncode(_buildStatusPayload()),
      );
      _appendLog('Status published to $statusTopic');

      await mqtt.publishString(
        topic: telemetryTopic,
        payload: jsonEncode(_buildTelemetryPayload()),
      );
      _appendLog('Telemetry published to $telemetryTopic');
    } catch (error, stackTrace) {
      _appendLog('Error: $error');
      Zone.current.handleUncaughtError(error, stackTrace);
    } finally {
      mqtt.disconnect();
      setState(() {
        _isSending = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('MQTT Test Utility'),
      ),
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Text(
                'Publish fake metadata/status/telemetry payloads to the broker.',
                style: TextStyle(fontSize: 16),
              ),
              const SizedBox(height: 16),
              _buildTextField(_brokerController, label: 'Broker address'),
              const SizedBox(height: 8),
              _buildTextField(_portController, label: 'Port', keyboardType: TextInputType.number),
              const SizedBox(height: 8),
              _buildTextField(_deviceNameController, label: 'Device name'),
              const SizedBox(height: 8),
              _buildTextField(_friendlyNameController, label: 'Friendly name'),
              const SizedBox(height: 8),
              _buildTextField(_hwIdController, label: 'Hardware ID / Registration suffix'),
              const SizedBox(height: 8),
              _buildTextField(_registrationCodeController, label: 'Registration code'),
              const SizedBox(height: 16),
              SizedBox(
                width: double.infinity,
                child: FilledButton.icon(
                  onPressed: _isSending ? null : _sendFakeData,
                  icon: const Icon(Icons.send),
                  label: Text(_isSending ? 'Sending…' : 'Send fake payloads'),
                ),
              ),
              const SizedBox(height: 16),
              Text('Logs (${_logs.length})', style: Theme.of(context).textTheme.titleMedium),
              const SizedBox(height: 8),
              Expanded(
                child: DecoratedBox(
                  decoration: BoxDecoration(
                    border: Border.all(color: Colors.teal.shade100),
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: _logs.isEmpty
                      ? const Center(child: Text('No logs yet.'))
                      : ListView.builder(
                          reverse: true,
                          padding: const EdgeInsets.all(12),
                          itemCount: _logs.length,
                          itemBuilder: (context, index) {
                            return Padding(
                              padding: const EdgeInsets.symmetric(vertical: 4),
                              child: Text(
                                _logs[index],
                                style: const TextStyle(fontFamily: 'monospace'),
                              ),
                            );
                          },
                        ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildTextField(
    TextEditingController controller, {
    required String label,
    TextInputType keyboardType = TextInputType.text,
  }) {
    return TextField(
      controller: controller,
      keyboardType: keyboardType,
      decoration: InputDecoration(
        labelText: label,
        border: const OutlineInputBorder(),
      ),
    );
  }
}
