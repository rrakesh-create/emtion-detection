import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:record/record.dart';
import 'package:path_provider/path_provider.dart';
import 'package:camera/camera.dart';
import 'package:image/image.dart' as img;

import 'api_service.dart';
import 'services/sync_service.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> with WidgetsBindingObserver {
  final AudioRecorder _audioRecorder = AudioRecorder();
  final ApiService _apiService = ApiService();
  
  bool _isRecording = false;
  String _statusText = "Ready";
  Map<String, dynamic>? _audioResult;
  String? _tempPath;

  final TextEditingController _ipController = TextEditingController();

  // Camera Support
  CameraController? _cameraController;
  List<CameraDescription>? _cameras;
  bool _isCameraInitialized = false;
  bool _isStreaming = false;
  Timer? _frameTimer;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _initRecorder();
    _initCamera();
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _cameraController?.dispose();
    _frameTimer?.cancel();
    _ipController.dispose();
    _audioRecorder.dispose();
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    // Handle camera resource release/resume if needed
    if (_cameraController == null || !_cameraController!.value.isInitialized) {
      return;
    }
    if (state == AppLifecycleState.inactive) {
      _cameraController?.dispose();
    } else if (state == AppLifecycleState.resumed) {
      if (_cameraController != null) {
        _initCamera();
      }
    }
  }

  Future<void> _initRecorder() async {
    final status = await Permission.microphone.request();
    if (status != PermissionStatus.granted) {
      setState(() => _statusText = "Microphone permission denied");
    }
  }

  Future<void> _initCamera() async {
    final status = await Permission.camera.request();
    if (status.isGranted) {
      try {
        _cameras = await availableCameras();
        if (_cameras != null && _cameras!.isNotEmpty) {
          // Use front camera if available, else first
          final camera = _cameras!.firstWhere(
            (c) => c.lensDirection == CameraLensDirection.front,
            orElse: () => _cameras!.first,
          );
          
          _cameraController = CameraController(
            camera,
            ResolutionPreset.medium, // 480p is enough for emotion
            enableAudio: false,
          );

          await _cameraController!.initialize();
          if (mounted) {
            setState(() {
              _isCameraInitialized = true;
            });
          }
        }
      } catch (e) {
        print("Camera Init Error: $e");
        setState(() => _statusText = "Camera Error: $e");
      }
    } else {
      setState(() => _statusText = "Camera permission denied");
    }
  }

  void _toggleStreaming() {
    if (_isStreaming) {
      _stopStreaming();
    } else {
      _startStreaming();
    }
  }

  void _startStreaming() {
    if (_frameTimer != null && _frameTimer!.isActive) return;
    
    setState(() => _isStreaming = true);
    // Send frame every 500ms (2 FPS)
    _frameTimer = Timer.periodic(const Duration(milliseconds: 500), (timer) async {
      if (!mounted) {
        timer.cancel();
        return;
      }
      if (_cameraController != null && 
          _cameraController!.value.isInitialized && 
          !_cameraController!.value.isTakingPicture) {
        await _captureAndSendFrame();
      }
    });
  }

  void _stopStreaming() {
    _frameTimer?.cancel();
    _frameTimer = null;
    if (mounted) {
      setState(() => _isStreaming = false);
    }
  }

  Future<void> _captureAndSendFrame() async {
    if (_cameraController == null || !_cameraController!.value.isInitialized) return;
    
    try {
      if (_cameraController!.value.isTakingPicture) return;
      
      // Capture JPEG
      final XFile file = await _cameraController!.takePicture();
      final bytes = await file.readAsBytes();
      
      // Send via SyncService
      if (mounted) {
        Provider.of<SyncService>(context, listen: false).sendFrame(bytes);
      }
    } catch (e) {
      print("Capture Error: $e");
      // If error indicates disposed controller, stop streaming
      if (e.toString().contains("Disposed")) {
         _stopStreaming();
         _initCamera(); // Try to re-init
      }
    }
  }

  Future<void> _startRecording() async {
    if (_ipController.text.isNotEmpty) {
      _apiService.setBaseUrl(_ipController.text);
      Provider.of<SyncService>(context, listen: false).setServerUrl(_ipController.text);
    }

    try {
      if (await _audioRecorder.hasPermission()) {
        final dir = await getTemporaryDirectory();
        _tempPath = '${dir.path}/audio_record.wav';
        
        // 16kHz, Mono, WAV
        const config = RecordConfig(
            encoder: AudioEncoder.wav,
            sampleRate: 16000,
            numChannels: 1
        );

        await _audioRecorder.start(config, path: _tempPath!);
        setState(() {
          _isRecording = true;
          _statusText = "Recording... (Release to send)";
          _audioResult = null;
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
        _statusText = "Analyzing Audio...";
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
          _audioResult = result;
          _statusText = "Analysis Complete";
        });
    } catch (e) {
      setState(() => _statusText = "Error: $e");
    }
  }

  @override
  Widget build(BuildContext context) {
    final syncService = Provider.of<SyncService>(context);

    return Scaffold(
      appBar: AppBar(title: const Text("MERS Client")),
      body: OrientationBuilder(
        builder: (context, orientation) {
          if (orientation == Orientation.portrait) {
             return SingleChildScrollView(
              child: Padding(
                padding: const EdgeInsets.all(16.0),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.stretch,
                  children: _buildChildren(syncService),
                ),
              ),
            );
          } else {
            // Landscape Layout
            return Row(
              children: [
                Expanded(
                  child: SingleChildScrollView(
                    child: Padding(
                      padding: const EdgeInsets.all(16.0),
                      child: Column(
                         crossAxisAlignment: CrossAxisAlignment.stretch,
                         children: [
                           _buildSyncStatus(syncService),
                           const SizedBox(height: 16),
                           _buildIpConfig(syncService),
                           const SizedBox(height: 16),
                           _buildCameraSection(syncService),
                         ],
                      ),
                    ),
                  ),
                ),
                const VerticalDivider(width: 1),
                Expanded(
                  child: SingleChildScrollView(
                    child: Padding(
                      padding: const EdgeInsets.all(16.0),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.stretch,
                        children: [
                          _buildRecordingSection(),
                          const SizedBox(height: 20),
                          _buildLocalResults(syncService),
                        ],
                      ),
                    ),
                  ),
                ),
              ],
            );
          }
        },
      ),
    );
  }

  List<Widget> _buildChildren(SyncService syncService) {
    return [
      _buildSyncStatus(syncService),
      const SizedBox(height: 16),
      _buildIpConfig(syncService),
      const SizedBox(height: 16),
      _buildCameraSection(syncService),
      const Divider(),
      _buildRecordingSection(),
      const SizedBox(height: 16),
      Text(_statusText, textAlign: TextAlign.center),
      const SizedBox(height: 20),
      _buildLocalResults(syncService),
    ];
  }

  Widget _buildSyncStatus(SyncService syncService) {
    return Card(
      color: syncService.isConnected ? Colors.green[100] : Colors.red[100],
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
            Text(
              syncService.isConnected ? "Connected to Server" : "Disconnected",
              style: const TextStyle(fontWeight: FontWeight.bold),
            ),
            if (syncService.isConnected)
              Text("Last Update: ${DateTime.now().toIso8601String().split('.').first}"),
          ],
        ),
      ),
    );
  }

  Widget _buildIpConfig(SyncService syncService) {
    return TextField(
      controller: _ipController,
      decoration: const InputDecoration(
        labelText: "Server IP (e.g., 192.168.1.X:8000)",
        border: OutlineInputBorder(),
        suffixIcon: Icon(Icons.save),
      ),
      onSubmitted: (value) {
        if (value.isNotEmpty) {
          _apiService.setBaseUrl(value);
          syncService.setServerUrl(value);
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(content: Text("Server IP Updated")),
          );
        }
      },
    );
  }

  Widget _buildCameraSection(SyncService syncService) {
    if (!_isCameraInitialized || _cameraController == null) {
      return Container(
        height: 200,
        color: Colors.grey[300],
        alignment: Alignment.center,
        child: const Text("Camera Initializing or Unavailable"),
      );
    }

    return Column(
      children: [
        AspectRatio(
          aspectRatio: _cameraController!.value.aspectRatio,
          child: Stack(
            fit: StackFit.expand,
            children: [
              CameraPreview(_cameraController!),
              
              if (_isStreaming)
                Positioned(
                  top: 10,
                  right: 10,
                  child: Container(
                    padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                    color: Colors.red,
                    child: const Text("LIVE INPUT", style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold)),
                  ),
                ),
                
              if (_isStreaming)
                Positioned(
                  bottom: 10,
                  left: 10,
                  right: 10,
                  child: Container(
                    padding: const EdgeInsets.all(8),
                    color: Colors.black54,
                    child: const Text(
                      "Sending data to PC...",
                      style: TextStyle(color: Colors.white, fontSize: 16),
                      textAlign: TextAlign.center,
                    ),
                  ),
                ),
            ],
          ),
        ),
        const SizedBox(height: 8),
        ElevatedButton.icon(
          onPressed: _toggleStreaming,
          icon: Icon(_isStreaming ? Icons.stop : Icons.play_arrow),
          label: Text(_isStreaming ? "Stop Streaming" : "Start Streaming"),
          style: ElevatedButton.styleFrom(
            backgroundColor: _isStreaming ? Colors.red : Colors.green,
            foregroundColor: Colors.white,
          ),
        ),
      ],
    );
  }

  Widget _buildRecordingSection() {
    return const SizedBox.shrink(); // Hiding Audio Recording for now to simplify UI
  }

  Widget _buildLocalResults(SyncService syncService) {
      return const SizedBox.shrink(); // Hiding Local Results
  }
}
