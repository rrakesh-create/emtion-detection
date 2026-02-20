import 'package:flutter/material.dart';
import 'home_screen.dart';

void main() {
  runApp(const MersApp());
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
