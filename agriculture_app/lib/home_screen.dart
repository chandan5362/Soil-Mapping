import 'package:agriculture_app/tabs/cropsTab.dart';
import 'package:agriculture_app/tabs/fertilizersTab.dart';
import 'package:agriculture_app/tabs/soilHealthTab.dart';
import 'package:agriculture_app/tabs/weatherTab.dart';
import 'package:flutter/material.dart';

class HomeScreen extends StatefulWidget {
  @override
  _HomeScreenState createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  int _selectedIdx=0;

  void _onItemTapped(int index) {
    setState(() {
      _selectedIdx = index;
    });
  }

  _tabBuilder(){
    switch(_selectedIdx){
      case 0:
        return WeatherTab();
      case 1:
        return SoilHealthTab();
      case 2:
        return CropsTab();
      case 3:
        return FertilizersTab();
      default:
        return Container(child: Text("No Such Tab",));
    }
  }
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text("Yep"),
      ),
      body: Center(
        child: _tabBuilder(),
      ),
      drawer: Drawer(
        child: ListView(shrinkWrap: true,
          children: [
            DrawerHeader(
              decoration: BoxDecoration(
                color: Color(0xffE4C5AF),
              ),
                child: Center(child: Container(child: Text("Username",style: TextStyle(color: Colors.black54),),))),
            ListTile(
              contentPadding: EdgeInsets.symmetric(vertical: 10,horizontal: 20),
              leading: Icon(Icons.home),
              title: Text("Home"),
              onTap: (){

              },
            ),
            ListTile(
              contentPadding: EdgeInsets.symmetric(vertical: 5,horizontal: 20),
              leading: Icon(Icons.location_city_outlined),
              title: Text("Govt. Schemes"),
              onTap: (){

              },
            ),
            ListTile(
              contentPadding: EdgeInsets.symmetric(vertical: 5,horizontal: 20),
              leading: Icon(Icons.logout),
              title: Text("Logout"),
              onTap: (){

              },
            ),
          ],
        ),
      ),
      floatingActionButtonLocation: FloatingActionButtonLocation.endFloat,
      floatingActionButton: FloatingActionButton(
        backgroundColor: Color(0xff1D263B),
        foregroundColor: Color(0xff92DCE5),
        child: Icon(Icons.location_on_sharp),
        tooltip: "Change location",
        onPressed: (){
          showDialog(
            context: context,
            builder: (context){
              return AlertDialog(
                title: Text("Change Location"),

                actions: [
                  TextButton(
                      onPressed: null,
                      child: Text("Change"),
                  ),
                ],
              );
            }
          );
        },
      ),
      floatingActionButtonAnimator: FloatingActionButtonAnimator.scaling,
      bottomNavigationBar: BottomAppBar(
        shape: CircularNotchedRectangle(),
        child: BottomNavigationBar(
          elevation: 10,
          type: BottomNavigationBarType.shifting,
          currentIndex: _selectedIdx,
          selectedItemColor: Colors.green,
          unselectedItemColor: Color(0xff5C6784),
          onTap: _onItemTapped,
          // unselectedLabelStyle: TextStyle(color: Colors.black),
          items: [
            BottomNavigationBarItem(
                icon: Icon(Icons.cloud),
                label: 'Weather'
            ),
            BottomNavigationBarItem(
                icon: Icon(Icons.bar_chart),
              label: 'Soil Health'
            ),
            BottomNavigationBarItem(
                icon: Icon(Icons.local_florist),
              label: 'Crops'
            ),
            BottomNavigationBarItem(
                icon: Icon(Icons.ac_unit_outlined),
                label: 'Fertilizers'
            ),
          ],
        ),
      ),
    );
  }
}
