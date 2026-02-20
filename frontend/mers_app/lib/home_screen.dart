import 'package:flutter/material.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:record/record.dart';
import 'package:path_provider/path_provider.dart';
import 'dart:io';
import 'api_service.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final AudioRecorder _audioRecorder = AudioRecorder();
  final ApiService _apiService = ApiService();
  
  bool _isRecording = false;
  String _statusText = "Ready";
  Map<String, dynamic>? _result;
  String? _tempPath;

  final TextEditingController _ipController = TextEditingController();

  @override
  void initState() {
    super.initState();
    _initRecorder();
  }

  Future<void> _initRecorder() async {
    final status = await Permission.microphone.request();
    if (status != PermissionStatus.granted) {
      setState(() => _statusText = "Microphone permission denied");
    }
  }

  Future<void> _startRecording() async {
    if (_ipController.text.isNotEmpty) {
      _apiService.setBaseUrl(_ipController.text);
    }

    try {
      if (await _audioRecorder.hasPermission()) {
        final dir = await getTemporaryDirectory();
        _tempPath = '${dir.path}/audio_record.wav';
        
        // 16kHz, Mono, WAV (Matches server requirements)
        const config = RecordConfig(
            encoder: AudioEncoder.wav,
            sampleRate: 16000,
            numChannels: 1
        );

        await _audioRecorder.start(config, path: _tempPath!);
        setState(() {
          _isRecording = true;
          _statusText = "Recording... (Release to send)";
          _result = null;
        });
      }
    } catch (e) {
      setState(() => _statusText = "Error starting: $e");
    }
  }

  Future<void> _stopRecording() async {
    try {
      final path = await _audioRecorder.stop();
      setState(() {
        _isRecording = false;
        _statusText = "Analyzing...";
      });

      if (path != null) {
        _sendAudio(path);
      }
    } catch (e) {
      setState(() => _statusText = "Error stopping: $e");
    }
  }

  Future<void> _sendAudio(String path) async {
    try {
        final result = await _apiService.analyzeAudio(path);
        setState(() {
          _result = result;
          _statusText = "Analysis Complete";
        });
    } catch (e) {
      setState(() => _statusText = "Error: $e");
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("MERS Client")),
      body: Padding(
        padding: const EdgeInsets.all(20.0),
        child: Column(
          children: [
            // IP Configuration
            TextField(
              controller: _ipController,
              decoration: const InputDecoration(
                labelText: "PC IP Address (e.g. 192.168.1.5)",
                border: OutlineInputBorder(),
              ),
              keyboardType: TextInputType.number,
            ),
            const SizedBox(height: 20),
            
            // Result Display
            Expanded(
              child: Center(
                child: _result == null
                    ? Text(_statusText, style: const TextStyle(fontSize: 18))
                    : _buildResultCard(),
              ),
            ),
            
            // Record Button
            GestureDetector(
              onLongPressStart: (_) => _startRecording(),
              onLongPressEnd: (_) => _stopRecording(),
              child: Container(
                width: 100,
                height: 100,
                decoration: BoxDecoration(
                  color: _isRecording ? Colors.red : Colors.blue,
                  shape: BoxShape.circle,
                  boxShadow: [
                    BoxShadow(color: Colors.black26, blurRadius: 10, spreadRadius: 2)
                  ]
                ),
                child: const Icon(Icons.mic, color: Colors.white, size: 50),
              ),
            ),
            const SizedBox(height: 20),
            const Text("Hold to Record", style: TextStyle(color: Colors.grey)),
          ],
        ),
      ),
    );
  }

  Color _hexToColor(String hexString) {
    final buffer = StringBuffer();
    if (hexString.length == 6 || hexString.length == 7) buffer.write('ff');
    buffer.write(hexString.replaceFirst('#', ''));
    return Color(int.parse(buffer.toString(), radix: 16));
  }

  Widget _buildResultCard() {
    final emotion = _result!['emotion'] ?? 'Unknown';
    final confidence = ((_result!['confidence'] ?? 0.0) * 100).toStringAsFixed(1);
    final colorHex = _result!['color'] ?? '#95A5A6'; // Default Neutral Gray
    final cardColor = _hexToColor(colorHex);
    
    // Parse Explanation (Handle both String and Map/Object)
    String explanationText = "";
    if (_result!['explanation'] is Map) {
      final expl = _result!['explanation'];
      explanationText = "👁️ ${expl['visual']}\n🎙️ ${expl['audio']}";
    } else {
      explanationText = _result!['explanation']?.toString() ?? "";
    }

    // Modality Weights
    String weightsText = "";
    if (_result!['modality_weights'] != null) {
        final w = _result!['modality_weights'];
        final aW = ((w['audio'] ?? 0) * 100).toInt();
        final vW = ((w['visual'] ?? 0) * 100).toInt();
        weightsText = "Audio: $aW% | Visual: $vW%";
    }

    return Card(
      elevation: 8,
      color: cardColor,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
      child: Padding(
        padding: const EdgeInsets.all(24.0),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Text(emotion, style: const TextStyle(fontSize: 40, fontWeight: FontWeight.bold, color: Colors.white)),
            const SizedBox(height: 10),
            Text("$confidence% Confidence", style: const TextStyle(fontSize: 20, color: Colors.white70)),
            if (weightsText.isNotEmpty) ...[
                const SizedBox(height: 5),
                Text(weightsText, style: const TextStyle(fontSize: 14, color: Colors.white60)),
            ],
            const Divider(color: Colors.white30),
            const SizedBox(height: 10),
            Text(explanationText, textAlign: TextAlign.center, style: const TextStyle(fontSize: 16, color: Colors.white)),
            const SizedBox(height: 20),
            Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                ElevatedButton.icon(
                  onPressed: () {}, // Feedback logic placeholder
                  icon: const Icon(Icons.thumb_up),
                  label: const Text("Correct"),
                  style: ElevatedButton.styleFrom(foregroundColor: cardColor, backgroundColor: Colors.white),
                ),
                const SizedBox(width: 10),
                IconButton(
                  onPressed: () {}, 
                  icon: const Icon(Icons.thumb_down, color: Colors.white70),
                )
              ],
            )
          ],
        ),
      ),
    );
  }
}
