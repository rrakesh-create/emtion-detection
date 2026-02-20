import 'dart:async';
import 'dart:convert';
import 'package:flutter/foundation.dart';
import 'package:web_socket_channel/web_socket_channel.dart';
import 'package:hive_flutter/hive_flutter.dart';
import 'package:http/http.dart' as http;

class SyncService extends ChangeNotifier {
  WebSocketChannel? _channel;
  WebSocketChannel? _visualChannel;
  bool _isConnected = false;
  Map<String, dynamic> _currentState = {};
  Map<String, dynamic> _localPrediction = {};
  Timer? _reconnectTimer;
  String _serverUrl = "ws://127.0.0.1:8001/ws/sync"; 
  String _visualUrl = "ws://127.0.0.1:8001/ws/mobile_visual";

  bool get isConnected => _isConnected;
  Map<String, dynamic> get currentState => _currentState;
  Map<String, dynamic> get localPrediction => _localPrediction;

  Future<void> init() async {
    await Hive.initFlutter();
    var box = await Hive.openBox('mers_cache');
    
    // Load cached state
    if (box.containsKey('last_state')) {
      _currentState = Map<String, dynamic>.from(box.get('last_state'));
      notifyListeners();
    }
    
    connect();
  }

  void setServerUrl(String url) {
    // url should be http://ip:port or just ip:port
    // We need to convert to ws://ip:port/ws/sync
    
    String cleanUrl = url.replaceAll("http://", "").replaceAll("https://", "");
    if (cleanUrl.endsWith("/")) cleanUrl = cleanUrl.substring(0, cleanUrl.length - 1);
    
    String wsUrl = "ws://$cleanUrl/ws/sync";
    String visUrl = "ws://$cleanUrl/ws/mobile_visual";
    
    if (_serverUrl != wsUrl) {
      _serverUrl = wsUrl;
      _visualUrl = visUrl;
      disconnect();
      connect();
    }
  }

  void disconnect() {
    _channel?.sink.close();
    _visualChannel?.sink.close();
    _channel = null;
    _visualChannel = null;
    _isConnected = false;
    _reconnectTimer?.cancel();
    notifyListeners();
  }

  void connect() {
    try {
      if (_channel != null) return;
      
      print("Connecting to $_serverUrl...");
      _channel = WebSocketChannel.connect(Uri.parse(_serverUrl));
      _isConnected = true;
      notifyListeners();

      _channel!.stream.listen(
        (message) {
          _handleMessage(message);
        },
        onDone: () {
          print("WebSocket Disconnected");
          _isConnected = false;
          _channel = null;
          notifyListeners();
          _scheduleReconnect();
        },
        onError: (error) {
          print("WebSocket Error: $error");
          _isConnected = false;
          notifyListeners();
          _scheduleReconnect();
        },
      );

      // Connect Visual Channel
      _connectVisual();

    } catch (e) {
      print("Connection Error: $e");
      _scheduleReconnect();
    }
  }

  void _connectVisual() {
    try {
        print("Connecting to Visual Stream: $_visualUrl");
        _visualChannel = WebSocketChannel.connect(Uri.parse(_visualUrl));
        _visualChannel!.stream.listen((message) {
            // Handle Visual Prediction Response
            try {
                final data = json.decode(message);
                _localPrediction = data;
                notifyListeners();
            } catch(e) {
                print("Visual Parse Error: $e");
            }
        });
    } catch (e) {
        print("Visual Connection Error: $e");
    }
  }

  void sendFrame(Uint8List bytes) {
    if (_visualChannel != null) {
        try {
            _visualChannel!.sink.add(bytes);
        } catch (e) {
            print("Send Frame Error: $e");
        }
    }
  }

  void _handleMessage(dynamic message) {
    try {
      final data = json.decode(message);
      _currentState = data;
      // Cache it
      Hive.box('mers_cache').put('last_state', _currentState);
      notifyListeners();
    } catch (e) {
      print("Error parsing message: $e");
    }
  }

  void _scheduleReconnect() {
    _reconnectTimer?.cancel();
    _reconnectTimer = Timer(const Duration(seconds: 5), () {
      print("Attempting reconnect...");
      connect();
    });
  }
  
  void updateSettings(Map<String, dynamic> settings) {
    if (_channel != null) {
      _channel!.sink.add(json.encode({"type": "settings", "data": settings}));
    }
  }

  @override
  void dispose() {
    _channel?.sink.close();
    _reconnectTimer?.cancel();
    super.dispose();
  }
}
