import 'package:flutter/material.dart';

class CropsTab extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    final _screenSize = MediaQuery.of(context).size;
    return Container(
      padding: EdgeInsets.all(10),
      child: Column(
        children: [
          Padding(
            padding: const EdgeInsets.all(8.0),
            child: Text("Recommended Crops",style: TextStyle(fontSize: 30),),
          ),
          SizedBox(
            height: _screenSize.height*0.7,
            child: ListView.separated(
              shrinkWrap: true,
                itemBuilder: (context,index){
                  return Card(
                    elevation: 10,
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(10),
                    ),
                    child: Container(
                      height: _screenSize.height*0.15,
                      padding: EdgeInsets.all(8),
                      child: Row(
                        children: [
                          Image.asset('assets/images/paddy-crop.jpg',height: _screenSize.height*1.5,width: _screenSize.width*0.3,),
                          Expanded(
                            child: Padding(
                              padding: const EdgeInsets.all(8.0),
                              child: Column(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: [
                                  Text("Rice Crop"),
                                  Text("Optimum Temperatures"),
                                  Text("Water Requirement"),
                                  Text("Nitrogen, Organic Compound in abundance")
                                ],
                              ),
                            ),
                          )
                        ],
                      ),
                    ),
                  );
                },
                separatorBuilder: (context,index){
                  return Padding(padding: EdgeInsets.all(10));
                },
                itemCount: 6),
          ),
        ],
      )
    );
  }
}
