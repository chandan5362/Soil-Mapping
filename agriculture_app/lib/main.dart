import 'package:agriculture_app/home_screen.dart';
import 'package:flutter/material.dart';

void main() {
  runApp(MyApp());
}


final Map<int, Color> color =
{

  50:Color(0xff519872).withOpacity(0.1),
  100: Color(0xff519872).withOpacity(0.2),
  200: Color(0xff519872).withOpacity(0.3),
  300: Color(0xff519872).withOpacity(0.4),
  400: Color(0xff519872).withOpacity(0.5),
  500: Color(0xff519872).withOpacity(0.6),
  600: Color(0xff519872).withOpacity(0.7),
  700: Color(0xff519872).withOpacity(0.8),
  800: Color(0xff519872).withOpacity(0.9),
  900: Color(0xff519872).withOpacity(1),
};
MaterialColor colorCustom = MaterialColor(0xFF519872, color);
class MyApp extends StatelessWidget {
  // This widget is the root of your application.
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Flutter Demo',
      theme: ThemeData(
        primarySwatch: Colors.green,
      ),
      debugShowCheckedModeBanner: false,
      home: HomeScreen(),
    );
  }
}

