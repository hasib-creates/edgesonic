import 'package:flutter/foundation.dart';
import 'package:mqtt_client/mqtt_client.dart';
import 'package:mqtt_client/mqtt_server_client.dart';

class MQTTService {
  MQTTService({
    required this.broker,
    this.clientId = 'edge-sonic-demo',
    this.port = 1883,
    this.keepAliveSeconds = 20,
  }) {
    _client = MqttServerClient.withPort(broker, clientId, port);
    _configureClient();
  }

  final String broker;
  final String clientId;
  final int port;
  final int keepAliveSeconds;

  late final MqttClient _client;

  MqttClient get client => _client;

  void _configureClient() {
    _client.logging(on: false);
    _client.keepAlivePeriod = keepAliveSeconds;
    _client.onConnected = () => debugPrint('MQTT connected');
    _client.onDisconnected = () => debugPrint('MQTT disconnected');
    _client.onSubscribed = (topic) => debugPrint('Subscribed to $topic');
    _client.connectionMessage =
        MqttConnectMessage().withClientIdentifier(clientId).startClean();
  }

  Future<void> connect({
    String? username,
    String? password,
  }) async {
    try {
      await _client.connect(username, password);
    } on Exception catch (error) {
      _client.disconnect();
      debugPrint('MQTT connect failed: $error');
      rethrow;
    }
  }

  Future<void> publishString({
    required String topic,
    required String payload,
    MqttQos qos = MqttQos.atLeastOnce,
  }) async {
    final builder = MqttClientPayloadBuilder()..addString(payload);
    _client.publishMessage(topic, qos, builder.payload!);
  }

  void disconnect() {
    _client.disconnect();
  }
}
