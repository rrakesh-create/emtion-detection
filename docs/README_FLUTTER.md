# MERS Mobile Client (Flutter)

This guide explains how to set up the Mobile App client for the Multimodal Emotion Recognition System (MERS).

## 1. Prerequisites
To build the mobile app, you need:
- **Flutter SDK**: [Install Flutter](https://docs.flutter.dev/get-started/install)
- **Android Studio** (for Android SDK and Emulator) or **VS Code** with Flutter extensions.
- An **Android Phone** (recommended) or Emulator.

## 2. Architecture Overview & FAQ
### How it works (Architecture)
- **PC (Server)**: The "Brain". It runs the Python/FastAPI backend, loads the AI models, and uses the **Laptop Webcam** to detect facial emotions.
- **Mobile (Client)**: The "Microphone". It runs the Flutter app, records **Audio**, and sends it to the PC.
- **Fusion**: The PC combines the *Audio from Mobile* with the *Video from Webcam* to produce a multimodal result.
- **Connection**: They communicate over **Local Wi-Fi**. Both devices must be on the *same network*.

### FAQ / Addressing Common Doubts
**Q: How do we convert the Python PC version to Mobile?**
A: We do **not** convert the Python code to Mobile. The Mobile app is a completely separate application written in **Dart/Flutter**. It only acts as a remote sensor (microphone) and display. The heavy AI processing stays on the PC.

**Q: The interfaces and display settings are different!**
A: Yes, Mobile and PC have different screens. 
- The **PC Interface** (if any) shows the webcam feed and server logs.
- The **Mobile Interface** (built with Flutter) is designed specifically for touchscreens. Flutter handles "responsive design" so it looks good on different phone sizes.

**Q: How do we process to APK and install?**
A: You don't "process" the Python code to APK. You build the **Flutter code** to APK using the command `flutter build apk`. Then you transfer that `.apk` file to your phone (via USB or download) and install it.

## 3. Create the Flutter Project
Open your terminal (outside the `mers` python folder) and run:

```bash
flutter create mers_mobile
cd mers_mobile
```

## 4. Update Dependencies (`pubspec.yaml`)
Open `pubspec.yaml` and add these dependencies under `dependencies:`:

```yaml
dependencies:
  flutter:
    sdk: flutter
  http: ^1.2.0
  flutter_sound: ^9.2.13
  permission_handler: ^11.3.0
  path_provider: ^2.1.2
```

Then run:
```bash
flutter pub get
```

## 5. Configure Permissions (`AndroidManifest.xml`)
Open `android/app/src/main/AndroidManifest.xml` and add these permissions before the `<application>` tag:

```xml
<uses-permission android:name="android.permission.INTERNET"/>
<uses-permission android:name="android.permission.RECORD_AUDIO"/>
<uses-permission android:name="android.permission.FOREGROUND_SERVICE"/>
```

## 6. App Code (`lib/main.dart`)
Replace the contents of `lib/main.dart` with the following code.
**IMPORTANT**: Update `SERVER_IP` with your PC's local IP address (e.g., `192.168.1.5`).

```dart
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter_sound/flutter_sound.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'package:path_provider/path_provider.dart';

// TODO: REPLACE WITH YOUR PC'S LOCAL IP ADDRESS
const String SERVER_IP = "192.168.1.X"; 
const String API_URL = "http://$SERVER_IP:8000/analyze_multimodal";

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'MERS Client',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
        useMaterial3: true,
      ),
      home: const MyHomePage(title: 'MERS Emotion Recorder'),
    );
  }
}

class MyHomePage extends StatefulWidget {
  const MyHomePage({super.key, required this.title});
  final String title;

  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  FlutterSoundRecorder? _recorder;
  bool _isRecording = false;
  String _status = "Ready";
  String _emotion = "--";
  String _explanation = "--";
  String _filePath = "";

  @override
  void initState() {
    super.initState();
    _recorder = FlutterSoundRecorder();
    _initRecorder();
  }

  Future<void> _initRecorder() async {
    await Permission.microphone.request();
    await _recorder!.openRecorder();
  }

  @override
  void dispose() {
    _recorder!.closeRecorder();
    super.dispose();
  }

  Future<void> _startRecording() async {
    Directory tempDir = await getTemporaryDirectory();
    _filePath = '${tempDir.path}/audio_record.wav';
    
    await _recorder!.startRecorder(
      toFile: _filePath,
      codec: Codec.pcm16WAV,
    );
    setState(() {
      _isRecording = true;
      _status = "Recording...";
    });
  }

  Future<void> _stopRecording() async {
    await _recorder!.stopRecorder();
    setState(() {
      _isRecording = false;
      _status = "Processing...";
    });
    await _uploadAudio();
  }

  Future<void> _uploadAudio() async {
    try {
      var request = http.MultipartRequest('POST', Uri.parse(API_URL));
      request.files.add(await http.MultipartFile.fromPath('file', _filePath));

      var response = await request.send();
      if (response.statusCode == 200) {
        var responseData = await response.stream.bytesToString();
        var json = jsonDecode(responseData);
        setState(() {
          _emotion = json['emotion'];
          _explanation = json['explanation'];
          _status = "Done";
        });
      } else {
        setState(() {
          _status = "Error: ${response.statusCode}";
        });
      }
    } catch (e) {
      setState(() {
        _status = "Connection Error: $e";
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text(widget.title)),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: <Widget>[
            Text('Status: $_status', style: Theme.of(context).textTheme.bodyLarge),
            SizedBox(height: 20),
            Text('Emotion:', style: Theme.of(context).textTheme.headlineSmall),
            Text(_emotion, style: Theme.of(context).textTheme.displayMedium?.copyWith(color: Colors.deepPurple)),
            SizedBox(height: 10),
            Padding(
              padding: const EdgeInsets.all(16.0),
              child: Text(_explanation, textAlign: TextAlign.center),
            ),
            SizedBox(height: 40),
            GestureDetector(
              onLongPressStart: (_) => _startRecording(),
              onLongPressEnd: (_) => _stopRecording(),
              child: Container(
                padding: EdgeInsets.all(30),
                decoration: BoxDecoration(
                  color: _isRecording ? Colors.red : Colors.blue,
                  shape: BoxShape.circle,
                ),
                child: Icon(
                  _isRecording ? Icons.mic : Icons.mic_none,
                  color: Colors.white,
                  size: 50,
                ),
              ),
            ),
            SizedBox(height: 10),
            Text("Hold to Record"),
          ],
        ),
      ),
    );
  }
}
```

## 7. Build and Install (`apk`)
1.  Connect your Android phone to your PC via USB.
2.  Enable **USB Debugging** in Developer Options on your phone.
3.  Run:
    ```bash
    flutter run --release
    ```
    This will build the app and install it directly on your phone.

4.  To generate an APK file to share:
    ```bash
    flutter build apk --release
    ```
    The APK will be at: `build/app/outputs/flutter-apk/app-release.apk`
    You can send this file to your phone and install it.

## 8. Troubleshooting
- **Connection Refused**: Ensure your PC firewall allows python/uvicorn connections on port 8000.
- **Wrong IP**: Run `ipconfig` (Windows) or `ifconfig` (Linux/Mac) to find your PC's IP address (e.g., Wireless LAN adapter Wi-Fi).
