

import 'package:flutter/cupertino.dart';

class Location extends ChangeNotifier{
  String locality;
  String state;
  String country;
  String latitude;
  String longitude;

  Location({this.locality,this.state,this.country,this.latitude,this.longitude});
}