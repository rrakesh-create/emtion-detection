import 'dart:convert';
import 'dart:io';
import 'package:http/http.dart' as http;

class ApiService {
  String baseUrl = 'http://192.168.1.X:8000'; // Default

  void setBaseUrl(String input) {
    String clean = input.trim().replaceAll(RegExp(r'https?://'), '');
    if (clean.contains(':')) {
       baseUrl = 'http://$clean';
    } else {
       baseUrl = 'http://$clean:8000';
    }
  }

  Future<Map<String, dynamic>> analyzeAudio(String filePath) async {
    final uri = Uri.parse('$baseUrl/analyze_audio');
    print("Connecting to: $uri");
    
    var request = http.MultipartRequest('POST', uri);
    request.files.add(await http.MultipartFile.fromPath('file', filePath));

    try {
      final streamedResponse = await request.send();
      final response = await http.Response.fromStream(streamedResponse);

      if (response.statusCode == 200) {
        // Decode and log for debugging
        final decoded = json.decode(response.body);
        print("API Response: $decoded");
        return decoded;
      } else {
        print("Server Error: ${response.body}");
        throw Exception('Failed to analyze audio: ${response.statusCode}');
      }
    } catch (e) {
      print("Connection Error: $e");
      throw Exception('Error connecting to $baseUrl: $e');
    }
  }

  Future<bool> checkHealth() async {
    try {
      final response = await http.get(Uri.parse('$baseUrl/health'));
      return response.statusCode == 200;
    } catch (e) {
      print('Health Check Failed: $e');
      return false;
    }
  }
}
