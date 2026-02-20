import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'home_screen.dart';
import 'services/sync_service.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  final syncService = SyncService();
  await syncService.init();
  
  runApp(
    MultiProvider(
      providers: [
        ChangeNotifierProvider.value(value: syncService),
      ],
      child: const MersApp(),
    ),
  );
}

class MersApp extends StatelessWidget {
  const MersApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'MERS Client',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
        useMaterial3: true,
      ),
      home: const HomeScreen(),
    );
  }
}
