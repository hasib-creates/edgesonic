import 'dart:async';

import 'package:tflite_flutter/tflite_flutter.dart';

/// Holds metadata that describes the loaded TensorFlow Lite model.
class ModelDescription {
  const ModelDescription({
    required this.inputShapes,
    required this.inputTypes,
    required this.outputShapes,
    required this.outputTypes,
  });

  final List<List<int>> inputShapes;
  final List<TensorType> inputTypes;
  final List<List<int>> outputShapes;
  final List<TensorType> outputTypes;
}

/// A summary of the warm-up inference run.
class WarmupResult {
  const WarmupResult({
    required this.elapsed,
    required this.outputSummaries,
  });

  final Duration elapsed;
  final List<WarmupOutputSummary> outputSummaries;
}

/// Captures a concise view into the model output tensors.
class WarmupOutputSummary {
  const WarmupOutputSummary({
    required this.index,
    required this.shape,
    required this.type,
    required this.sampleValues,
  });

  final int index;
  final List<int> shape;
  final TensorType type;
  final List<String> sampleValues;
}

/// Service that wraps TensorFlow Lite interpreter lifecycle management.
class TFLiteService {
  TFLiteService({this.assetPath = 'assets/models/demo_model.tflite'});

  final String assetPath;

  Interpreter? _interpreter;

  bool get isLoaded => _interpreter != null;

  Future<void> loadModel() async {
    if (_interpreter != null) {
      return;
    }

    try {
      final options = InterpreterOptions()..threads = 2;
      _interpreter = await Interpreter.fromAsset(assetPath, options: options);
    } on Object {
      // Clean up any partial interpreter when load fails.
      close();
      rethrow;
    }
  }

  ModelDescription describe() {
    final interpreter = _requireInterpreter();
    final inputTensors = interpreter.getInputTensors();
    final outputTensors = interpreter.getOutputTensors();

    return ModelDescription(
      inputShapes: inputTensors.map((tensor) => List<int>.from(tensor.shape)).toList(growable: false),
      inputTypes: inputTensors.map((tensor) => tensor.type).toList(growable: false),
      outputShapes: outputTensors.map((tensor) => List<int>.from(tensor.shape)).toList(growable: false),
      outputTypes: outputTensors.map((tensor) => tensor.type).toList(growable: false),
    );
  }

  Future<WarmupResult> runWarmupInference() async {
    final interpreter = _requireInterpreter();

    final inputTensors = interpreter.getInputTensors();
    final outputTensors = interpreter.getOutputTensors();

    final inputs = List<Object>.generate(
      inputTensors.length,
      (i) => _createZeroFilledTensor(inputTensors[i]),
      growable: false,
    );

    final rawOutputs = <int, Object>{};
    for (var i = 0; i < outputTensors.length; i++) {
      rawOutputs[i] = _createZeroFilledTensor(outputTensors[i]);
    }

    final stopwatch = Stopwatch()..start();
    interpreter.runForMultipleInputs(inputs, rawOutputs);
    stopwatch.stop();

    final summaries = <WarmupOutputSummary>[];
    for (var i = 0; i < outputTensors.length; i++) {
      final tensor = outputTensors[i];
      final tensorData = rawOutputs[i]!;
      summaries.add(
        WarmupOutputSummary(
          index: i,
          shape: List<int>.from(tensor.shape),
          type: tensor.type,
          sampleValues: _collectSampleValues(tensorData),
        ),
      );
    }

    final micros = interpreter.lastNativeInferenceDurationMicroSeconds;
    final elapsed = micros > 0 ? Duration(microseconds: micros) : stopwatch.elapsed;

    return WarmupResult(
      elapsed: elapsed,
      outputSummaries: summaries,
    );
  }

  void close() {
    _interpreter?.close();
    _interpreter = null;
  }

  Interpreter _requireInterpreter() {
    final interpreter = _interpreter;
    if (interpreter == null) {
      throw StateError('Interpreter is not loaded. Call loadModel() first.');
    }
    return interpreter;
  }

  Object _createZeroFilledTensor(Tensor tensor) {
    return _createZeroFilledShape(tensor.shape, tensor.type);
  }

  Object _createZeroFilledShape(List<int> shape, TensorType type) {
    if (shape.isEmpty) {
      return _zeroValue(type);
    }
    if (shape.length == 1) {
      return List.generate(shape.first, (_) => _zeroValue(type), growable: false);
    }
    final subShape = shape.sublist(1);
    return List.generate(
      shape.first,
      (_) => _createZeroFilledShape(subShape, type),
      growable: false,
    );
  }

  Object _zeroValue(TensorType type) {
    switch (type) {
      case TensorType.float32:
      case TensorType.float64:
      case TensorType.float16:
        return 0.0;
      case TensorType.int4:
      case TensorType.int8:
      case TensorType.int16:
      case TensorType.int32:
      case TensorType.int64:
      case TensorType.uint8:
      case TensorType.uint16:
      case TensorType.uint32:
      case TensorType.uint64:
        return 0;
      case TensorType.boolean:
        return false;
      default:
        throw UnsupportedError(
          'Tensor type $type is not supported by the warm-up helper. Provide custom inference instead.',
        );
    }
  }

  List<String> _collectSampleValues(Object data, {int limit = 8}) {
    final samples = <String>[];

    void read(Object value) {
      if (samples.length >= limit) {
        return;
      }
      if (value is List) {
        for (final element in value) {
          if (samples.length >= limit) {
            break;
          }
          read(element);
        }
      } else {
        samples.add(value.toString());
      }
    }

    read(data);
    return samples;
  }
}
