// import 'package:charts_flutter/flutter.dart';
import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';

class WeatherTab extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    final _screenSize = MediaQuery.of(context).size;
    return Container(
      padding: EdgeInsets.only(top: 10),
      color: Colors.white,
      child: Stack(
        children: [
          Container(
            padding: EdgeInsets.all(10),
            child: Column(
              children: [
                Container(
                  padding: EdgeInsets.all(10),
                  child: Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text("Today, 20 Feb",style: TextStyle(fontSize: 15,fontWeight: FontWeight.w400),),
                          Text("Location Name",style: TextStyle(fontSize: 25),),
                        ],
                      ),
                      Icon(Icons.location_on_sharp,size: 40,)
                    ],
                  ),
                ),
                Image.asset('assets/images/weather-sunny-sun-cloudy.png',
                  height: _screenSize.height*0.25,
                  width: _screenSize.height*0.25,
                  fit: BoxFit.cover,
                ),
                Padding(
                  padding: const EdgeInsets.only(bottom:8.0),
                  child: Text("Cloudy",style: TextStyle(color: Colors.black54,fontSize: 30),),
                ),
                Card(
                  elevation: 10,
                  color: Colors.green,
                  child: Container(
                      padding: EdgeInsets.all(5),
                      height: 100,
                      child: Row(
                        mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                        children: [
                          Container(
                            width: _screenSize.width * 0.25,
                            child: Column(
                              mainAxisAlignment: MainAxisAlignment.center,
                              children: [Text("Wind",style: TextStyle(color: Colors.white,fontSize: 17),), Text("200",style: TextStyle(color: Colors.white,fontSize: 15),)],
                            ),
                          ),
                          Container(
                            margin: EdgeInsets.all(5),
                            height: 60,
                            width: 2,
                            decoration: BoxDecoration(
                              // border: Border.all(color: Colors.white),
                              color: Colors.white,
                            ),
                          ),
                          Container(
                            width: _screenSize.width * 0.25,
                            child: Column(
                              mainAxisAlignment: MainAxisAlignment.center,
                              children: [
                                Text("Temparature",style: TextStyle(color: Colors.white,fontSize: 17),),
                                Text("20 C",style: TextStyle(color: Colors.white,fontSize: 15),)
                              ],
                            ),
                          ),
                          Container(
                            margin: EdgeInsets.all(5),
                            height: 60,
                            width: 2,
                            decoration: BoxDecoration(
                              // border: Border.all(color: Colors.white),
                              color: Colors.white,
                            ),
                          ),
                          Container(
                            width: _screenSize.width * 0.25,
                            child: Column(
                              mainAxisAlignment: MainAxisAlignment.center,
                              children: [Text("Humidity",style: TextStyle(color: Colors.white,fontSize: 17),), Text("25%",style: TextStyle(color: Colors.white,fontSize: 11),)],
                            ),
                          )
                        ],
                      )),
                ),
              ],
            ),
          ),
          DraggableScrollableSheet(
            initialChildSize: 0.3,
            minChildSize: 0.3,
            maxChildSize: 0.85,
            builder: (BuildContext context, scrollController) {
              return Card(
                elevation: 10,
                color: Color(0xff5C6784),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.only(
                      topLeft: Radius.circular(10),
                      topRight: Radius.circular(10)),
                ),
                child: Container(
                  margin: EdgeInsets.all(10),
                  padding: EdgeInsets.symmetric(horizontal: 15, vertical: 10),
                  child: ListView(
                    // shrinkWrap: true,
                    controller: scrollController,
                    children: [
                      Icon(
                        Icons.maximize_rounded,
                        size: 40,
                        color: Colors.white,
                      ),
                      SizedBox(
                        height: _screenSize.height*0.1,
                        child: ListView.separated(
                          shrinkWrap: true,
                          physics: ClampingScrollPhysics(),
                          itemCount: 7,
                            scrollDirection: Axis.horizontal,
                            itemBuilder: (context,index){
                             return Container(
                               child: Column(
                                 children: [
                                   Icon(Icons.cloud,color: Colors.white70,),
                                   Text("Cloudy",style: TextStyle(color: Colors.white70,fontSize: 10),),
                                 ],
                               ),
                             );
                            },
                            separatorBuilder: (context,index){
                            return Padding(
                              padding: EdgeInsets.all(10),
                            );
                            }, ),
                      ),
                      ListView.separated(
                          separatorBuilder: (context,index){
                            return Divider(
                              color: Colors.white30,
                              thickness: 2,
                              indent: 10,
                            );
                          },
                          itemCount: 7,
                          shrinkWrap: true,
                          physics: ClampingScrollPhysics(),
                          itemBuilder: (context,index){
                            return Container(
                              padding: EdgeInsets.all(8),
                              child: Column(
                                mainAxisSize: MainAxisSize.min,
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: [
                                  Text("Saturday,24 Feb",style: TextStyle(color: Colors.white,fontSize: 13),),
                                  Container(
                                      padding: EdgeInsets.all(10),
                                      child: Row(
                                        mainAxisAlignment: MainAxisAlignment.spaceAround,
                                        children: [
                                          Container(
                                            width: _screenSize.width * 0.1,
                                            child: Column(
                                              mainAxisAlignment: MainAxisAlignment.center,
                                              children: [Text("Wind",style: TextStyle(color: Colors.white,fontSize: 13),), Text("200",style: TextStyle(color: Colors.white,fontSize: 11),)],
                                            ),
                                          ),
                                          Container(
                                            height: 30,
                                            width: 1,
                                            decoration: BoxDecoration(
                                              // border: Border.all(color: Colors.white),
                                              color: Colors.white,
                                            ),
                                          ),
                                          Container(
                                            width: _screenSize.width * 0.1,
                                            child: Column(
                                              mainAxisAlignment: MainAxisAlignment.center,
                                              children: [
                                                Text("Temp",style: TextStyle(color: Colors.white,fontSize: 13),),
                                                Text("20 C",style: TextStyle(color: Colors.white,fontSize: 11),)
                                              ],
                                            ),
                                          ),
                                          Container(
                                            height: 30,
                                            width: 1,
                                            decoration: BoxDecoration(
                                              // border: Border.all(color: Colors.white),
                                              color: Colors.white,
                                            ),
                                          ),
                                          Container(
                                            width: _screenSize.width * 0.25,
                                            child: Column(
                                              mainAxisAlignment: MainAxisAlignment.center,
                                              children: [Text("Humidity",style: TextStyle(color: Colors.white,fontSize: 13),), Text("25%",style: TextStyle(color: Colors.white,fontSize: 15),)],
                                            ),
                                          )
                                        ],
                                      )),
                                ],
                              ),
                            );
                          })
                    ],
                  ),
                ),
              );
            },
          ),
        ],
      ),
    );
  }
}
