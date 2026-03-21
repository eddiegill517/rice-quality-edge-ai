import 'dart:io';
import 'dart:math';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image_picker/image_picker.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;

void main() => runApp(const RiceQualityApp());

class RiceQualityApp extends StatelessWidget {
  const RiceQualityApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Rice Quality Inspector',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF2E7D32),
        ),
        useMaterial3: true,
      ),
      home: const InspectorScreen(),
      debugShowCheckedModeBanner: false,
    );
  }
}

class InspectorScreen extends StatefulWidget {
  const InspectorScreen({super.key});

  @override
  State<InspectorScreen> createState() => _InspectorScreenState();
}

class _InspectorScreenState extends State<InspectorScreen> {
  Interpreter? _interpreter;
  File? _selectedImage;
  String _result = '';
  double _confidence = 0;
  double _inferenceTime = 0;
  bool _isProcessing = false;
  Map<String, double>? _allPredictions;
  String _debugInfo = '';
  bool _isNHWC = false;

  final List<String> _classNames = [
    'Broken Rice',
    'Chalky Rice',
    'Foreign Object',
    'Head Rice',
    'Unhulled Rice',
  ];

  static const int inputSize = 128;
  static const List<double> mean = [0.485, 0.456, 0.406];
  static const List<double> std = [0.229, 0.224, 0.225];

  @override
  void initState() {
    super.initState();
    _loadModel();
  }

  Future<void> _loadModel() async {
    try {
      final data = await rootBundle.load('assets/model.tflite');
      final buffer = data.buffer.asUint8List();
      _interpreter = Interpreter.fromBuffer(buffer);

      var inputTensor = _interpreter!.getInputTensor(0);
      var inputShape = inputTensor.shape;
      var inputType = inputTensor.type;

      if (inputShape.length == 4 && inputShape[3] == 3) {
        _isNHWC = true;
      } else {
        _isNHWC = false;
      }

      var outputShape = _interpreter!.getOutputTensor(0).shape;

      setState(() {
        _debugInfo =
            'Input: $inputShape ($inputType) ${_isNHWC ? "NHWC" : "NCHW"}\n'
            'Output: $outputShape';
      });

      debugPrint('Model loaded successfully');
      debugPrint('Input shape: $inputShape type: $inputType');
      debugPrint('Format: ${_isNHWC ? "NHWC" : "NCHW"}');
      debugPrint('Output shape: $outputShape');
    } catch (e) {
      debugPrint('Error loading model: $e');
      setState(() {
        _debugInfo = 'Error: $e';
      });
    }
  }

  Future<void> _pickImage(ImageSource source) async {
    final picker = ImagePicker();
    final picked = await picker.pickImage(
      source: source,
      maxWidth: 1024,
      maxHeight: 1024,
    );

    if (picked != null) {
      setState(() {
        _selectedImage = File(picked.path);
        _result = '';
        _confidence = 0;
        _allPredictions = null;
      });
      await _runInference();
    }
  }

  Float32List _preprocessNHWC(img.Image image) {
    final Float32List input = Float32List(1 * inputSize * inputSize * 3);
    int idx = 0;

    for (int y = 0; y < inputSize; y++) {
      for (int x = 0; x < inputSize; x++) {
        final pixel = image.getPixel(x, y);

        double r = pixel.r.toDouble();
        double g = pixel.g.toDouble();
        double b = pixel.b.toDouble();

        input[idx++] = ((r / 255.0) - mean[0]) / std[0];
        input[idx++] = ((g / 255.0) - mean[1]) / std[1];
        input[idx++] = ((b / 255.0) - mean[2]) / std[2];
      }
    }

    return input;
  }

  Float32List _preprocessNCHW(img.Image image) {
    final Float32List input = Float32List(1 * 3 * inputSize * inputSize);

    for (int c = 0; c < 3; c++) {
      for (int y = 0; y < inputSize; y++) {
        for (int x = 0; x < inputSize; x++) {
          final pixel = image.getPixel(x, y);

          double value;
          if (c == 0) {
            value = pixel.r.toDouble();
          } else if (c == 1) {
            value = pixel.g.toDouble();
          } else {
            value = pixel.b.toDouble();
          }

          int idx = c * inputSize * inputSize + y * inputSize + x;
          input[idx] = ((value / 255.0) - mean[c]) / std[c];
        }
      }
    }

    return input;
  }

  Future<void> _runInference() async {
    if (_interpreter == null || _selectedImage == null) return;

    setState(() => _isProcessing = true);

    try {
      final bytes = await _selectedImage!.readAsBytes();
      final decoded = img.decodeImage(bytes);
      if (decoded == null) {
        setState(() {
          _result = 'Failed to decode image';
          _isProcessing = false;
        });
        return;
      }

      final resized = img.copyResize(
        decoded,
        width: inputSize,
        height: inputSize,
        interpolation: img.Interpolation.linear,
      );

      Float32List inputData;
      List<int> inputShape;

      if (_isNHWC) {
        inputData = _preprocessNHWC(resized);
        inputShape = [1, inputSize, inputSize, 3];
      } else {
        inputData = _preprocessNCHW(resized);
        inputShape = [1, 3, inputSize, inputSize];
      }

      debugPrint(
          'First 10 input values: ${inputData.sublist(0, 10).map((v) => v.toStringAsFixed(3)).toList()}');

      var outputBuffer =
          List.filled(_classNames.length, 0.0).reshape([1, _classNames.length]);

      var inputTensor = inputData.reshape(inputShape);

      final stopwatch = Stopwatch()..start();
      _interpreter!.run(inputTensor, outputBuffer);
      stopwatch.stop();

      List<double> logits = List<double>.from(outputBuffer[0]);

      debugPrint('Raw logits: $logits');

      List<double> probs = _softmax(logits);
      debugPrint('Probabilities: $probs');

      int maxIdx = 0;
      double maxProb = probs[0];
      Map<String, double> predictions = {};

      for (int i = 0; i < probs.length; i++) {
        predictions[_classNames[i]] = probs[i];
        if (probs[i] > maxProb) {
          maxProb = probs[i];
          maxIdx = i;
        }
      }

      var sortedPredictions = Map.fromEntries(
        predictions.entries.toList()
          ..sort((a, b) => b.value.compareTo(a.value)),
      );

      setState(() {
        _result = _classNames[maxIdx];
        _confidence = maxProb;
        _inferenceTime = stopwatch.elapsedMicroseconds / 1000.0;
        _allPredictions = sortedPredictions;
        _isProcessing = false;
      });
    } catch (e, st) {
      debugPrint('Inference error: $e');
      debugPrint('Stack trace: $st');
      setState(() {
        _result = 'Error: $e';
        _isProcessing = false;
      });
    }
  }

  List<double> _softmax(List<double> logits) {
    double maxLogit = logits.reduce(max);
    List<double> expValues =
        logits.map((l) => exp(l - maxLogit)).toList();
    double sumExp = expValues.reduce((a, b) => a + b);
    return expValues.map((e) => e / sumExp).toList();
  }

  Color _getConfidenceColor(double confidence) {
    if (confidence >= 0.7) return const Color(0xFF2E7D32);
    if (confidence >= 0.4) return const Color(0xFFF57F17);
    return const Color(0xFFC62828);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFFF5F5F0),
      appBar: AppBar(
        title: const Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Text('🌾 ', style: TextStyle(fontSize: 24)),
            Text(
              'Rice Quality Inspector',
              style: TextStyle(fontWeight: FontWeight.bold),
            ),
          ],
        ),
        centerTitle: true,
        backgroundColor: const Color(0xFF2E7D32),
        foregroundColor: Colors.white,
        elevation: 2,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          children: [
            Container(
              padding:
                  const EdgeInsets.symmetric(horizontal: 14, vertical: 7),
              decoration: BoxDecoration(
                color: _interpreter != null
                    ? Colors.green.shade50
                    : Colors.red.shade50,
                borderRadius: BorderRadius.circular(20),
                border: Border.all(
                  color: _interpreter != null
                      ? Colors.green.shade300
                      : Colors.red.shade300,
                ),
              ),
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Icon(
                    _interpreter != null
                        ? Icons.check_circle
                        : Icons.error,
                    size: 16,
                    color: _interpreter != null
                        ? Colors.green.shade700
                        : Colors.red.shade700,
                  ),
                  const SizedBox(width: 6),
                  Text(
                    _interpreter != null
                        ? 'TFLite model loaded • On-device inference'
                        : 'Model not loaded',
                    style: TextStyle(
                      fontSize: 12,
                      color: _interpreter != null
                          ? Colors.green.shade700
                          : Colors.red.shade700,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 16),
            Container(
              height: 300,
              width: double.infinity,
              decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.circular(16),
                border: Border.all(color: Colors.grey.shade300),
                boxShadow: [
                  BoxShadow(
                    color: Colors.black.withOpacity(0.05),
                    blurRadius: 10,
                    offset: const Offset(0, 2),
                  ),
                ],
              ),
              child: _selectedImage != null
                  ? ClipRRect(
                      borderRadius: BorderRadius.circular(16),
                      child: Image.file(
                        _selectedImage!,
                        fit: BoxFit.contain,
                      ),
                    )
                  : Center(
                      child: Column(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          Icon(Icons.grain,
                              size: 64, color: Colors.grey.shade400),
                          const SizedBox(height: 12),
                          Text(
                            'Take or select a photo of rice grains',
                            style: TextStyle(
                              color: Colors.grey.shade500,
                              fontSize: 15,
                            ),
                          ),
                        ],
                      ),
                    ),
            ),
            const SizedBox(height: 16),
            Row(
              children: [
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: () => _pickImage(ImageSource.camera),
                    icon: const Icon(Icons.camera_alt),
                    label: const Text('Camera'),
                    style: ElevatedButton.styleFrom(
                      padding: const EdgeInsets.symmetric(vertical: 14),
                      backgroundColor: const Color(0xFF2E7D32),
                      foregroundColor: Colors.white,
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(12),
                      ),
                    ),
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: OutlinedButton.icon(
                    onPressed: () => _pickImage(ImageSource.gallery),
                    icon: const Icon(Icons.photo_library),
                    label: const Text('Gallery'),
                    style: OutlinedButton.styleFrom(
                      padding: const EdgeInsets.symmetric(vertical: 14),
                      foregroundColor: const Color(0xFF2E7D32),
                      side: const BorderSide(color: Color(0xFF2E7D32)),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(12),
                      ),
                    ),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 20),
            if (_isProcessing)
              const Padding(
                padding: EdgeInsets.all(32),
                child: CircularProgressIndicator(
                    color: Color(0xFF2E7D32)),
              )
            else if (_result.isNotEmpty &&
                !_result.startsWith('Error')) ...[
              Container(
                width: double.infinity,
                padding: const EdgeInsets.all(24),
                decoration: BoxDecoration(
                  gradient: LinearGradient(
                    begin: Alignment.topLeft,
                    end: Alignment.bottomRight,
                    colors: [
                      _getConfidenceColor(_confidence),
                      _getConfidenceColor(_confidence).withOpacity(0.8),
                    ],
                  ),
                  borderRadius: BorderRadius.circular(16),
                  boxShadow: [
                    BoxShadow(
                      color: _getConfidenceColor(_confidence)
                          .withOpacity(0.3),
                      blurRadius: 12,
                      offset: const Offset(0, 4),
                    ),
                  ],
                ),
                child: Column(
                  children: [
                    const Text(
                      'Classification Result',
                      style:
                          TextStyle(color: Colors.white70, fontSize: 14),
                    ),
                    const SizedBox(height: 8),
                    Text(
                      _result,
                      style: const TextStyle(
                        color: Colors.white,
                        fontSize: 28,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    const SizedBox(height: 6),
                    Text(
                      '${(_confidence * 100).toStringAsFixed(1)}% confidence',
                      style: const TextStyle(
                          color: Colors.white70, fontSize: 16),
                    ),
                    const SizedBox(height: 10),
                    Container(
                      padding: const EdgeInsets.symmetric(
                          horizontal: 12, vertical: 4),
                      decoration: BoxDecoration(
                        color: Colors.white.withOpacity(0.2),
                        borderRadius: BorderRadius.circular(12),
                      ),
                      child: Text(
                        '⚡ ${_inferenceTime.toStringAsFixed(1)} ms',
                        style: const TextStyle(
                            color: Colors.white70, fontSize: 13),
                      ),
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 16),
              if (_allPredictions != null)
                Container(
                  width: double.infinity,
                  padding: const EdgeInsets.all(20),
                  decoration: BoxDecoration(
                    color: Colors.white,
                    borderRadius: BorderRadius.circular(16),
                    boxShadow: [
                      BoxShadow(
                        color: Colors.black.withOpacity(0.05),
                        blurRadius: 10,
                        offset: const Offset(0, 2),
                      ),
                    ],
                  ),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Text(
                        'All Classes',
                        style: TextStyle(
                          fontWeight: FontWeight.bold,
                          fontSize: 17,
                        ),
                      ),
                      const SizedBox(height: 14),
                      ..._allPredictions!.entries.map(
                        (e) => Padding(
                          padding: const EdgeInsets.only(bottom: 12),
                          child: Column(
                            crossAxisAlignment:
                                CrossAxisAlignment.start,
                            children: [
                              Row(
                                mainAxisAlignment:
                                    MainAxisAlignment.spaceBetween,
                                children: [
                                  Text(
                                    e.key,
                                    style: TextStyle(
                                      fontWeight: e.key == _result
                                          ? FontWeight.bold
                                          : FontWeight.normal,
                                      fontSize: 14,
                                    ),
                                  ),
                                  Text(
                                    '${(e.value * 100).toStringAsFixed(1)}%',
                                    style: TextStyle(
                                      fontWeight: e.key == _result
                                          ? FontWeight.bold
                                          : FontWeight.normal,
                                      color: e.key == _result
                                          ? _getConfidenceColor(
                                              _confidence)
                                          : Colors.grey.shade600,
                                    ),
                                  ),
                                ],
                              ),
                              const SizedBox(height: 4),
                              ClipRRect(
                                borderRadius:
                                    BorderRadius.circular(4),
                                child: LinearProgressIndicator(
                                  value: e.value,
                                  minHeight: 6,
                                  backgroundColor:
                                      Colors.grey.shade200,
                                  valueColor:
                                      AlwaysStoppedAnimation(
                                    e.key == _result
                                        ? _getConfidenceColor(
                                            _confidence)
                                        : Colors.grey.shade400,
                                  ),
                                ),
                              ),
                            ],
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
            ] else if (_result.startsWith('Error')) ...[
              Container(
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: Colors.red.shade50,
                  borderRadius: BorderRadius.circular(12),
                ),
                child: Text(_result,
                    style: TextStyle(color: Colors.red.shade700)),
              ),
            ],
            if (_debugInfo.isNotEmpty) ...[
              const SizedBox(height: 16),
              Container(
                width: double.infinity,
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: Colors.grey.shade100,
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Text(
                  _debugInfo,
                  style: TextStyle(
                    fontSize: 11,
                    fontFamily: 'monospace',
                    color: Colors.grey.shade600,
                  ),
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }

  @override
  void dispose() {
    _interpreter?.close();
    super.dispose();
  }
}