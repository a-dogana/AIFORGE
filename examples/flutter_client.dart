import 'dart:convert';
import 'package:http/http.dart' as http;

/// NOTE: On Android Emulator, use 'http://10.0.2.2:8000' instead of localhost.
/// On physical iOS/Android devices, use the local IP address of your dev machine.
class AiforgeClient {
  final String baseUrl;

  AiforgeClient({this.baseUrl = 'http://127.0.0.1:8000'});

  Future<bool> checkReady() async {
    final response = await http.get(Uri.parse('$baseUrl/ready'));
    return response.statusCode == 200;
  }

  Future<Map<String, dynamic>> getModelMetadata() async {
    final response = await http.get(Uri.parse('$baseUrl/v1/model'));
    if (response.statusCode == 200) {
      return jsonDecode(response.body);
    }
    throw Exception('Failed to load metadata');
  }

  Future<List<dynamic>> predict(List<List<double>> data) async {
    final response = await http.post(
      Uri.parse('$baseUrl/v1/predict'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({'data': data}),
    ).timeout(const Duration(seconds: 5));

    if (response.statusCode == 200) {
      final jsonResponse = jsonDecode(response.body);
      return jsonResponse['predictions'];
    } else {
      throw Exception('Failed to predict: ${response.body}');
    }
  }
}

void main() async {
  final client = AiforgeClient();
  
  print('Checking if AIFORGE Server is ready...');
  try {
    bool isReady = await client.checkReady();
    if (isReady) {
      print('Server is READY.');
      
      final meta = await client.getModelMetadata();
      print('Model: ${meta["experiment_name"]}');
      
      // Dummy data (e.g. 64 zeros for Digits MLP)
      List<double> dummyRow = List.filled(64, 0.0);
      print('Sending prediction request...');
      
      final predictions = await client.predict([dummyRow]);
      print('Predictions: $predictions');
    }
  } catch (e) {
    print('Error communicating with AIFORGE Server: $e');
  }
}
