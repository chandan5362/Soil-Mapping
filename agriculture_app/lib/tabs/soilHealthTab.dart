import 'package:flutter/material.dart';
import 'package:charts_flutter/flutter.dart' as charts;

class SoilHealthTab extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    var data = [
      NutrientContent('Nitrogen', 30, Colors.red),
      NutrientContent('Pottasium', 15, Colors.yellow),
      NutrientContent('Sodium', 42, Colors.blue),
      NutrientContent('Organic C.', 60, Colors.green),
    ];

    var series = [
      charts.Series(
        domainFn: (NutrientContent nutrientContent, _) => nutrientContent.nutrient,
        measureFn: (NutrientContent nutrientContent, _) => nutrientContent.content,
        colorFn: (NutrientContent nutrientContent, _) => nutrientContent.color,
        id: 'Content',
        data: data,
      ),
    ];
    var chart = charts.BarChart(
      series,
      animate: true,
    );
    var pieChart=charts.PieChart(
      series,
      animate: true,
        defaultRenderer: new charts.ArcRendererConfig(arcWidth: 60,arcRendererDecorators: [
          new charts.ArcLabelDecorator(
              labelPosition: charts.ArcLabelPosition.auto)
        ]));
    return Container(
      child: Column(
        children: [
          Container(
            height: 300,
            width: 350,
            child: Padding(
              padding: const EdgeInsets.all(8.0),
              child: chart,
            ),
          ),
          Container(
            height: 300,
            width: 300,
            child: Padding(
              padding: const EdgeInsets.all(8.0),
              child: pieChart,
            ),
          ),
        ],
      ),
    );
  }
}

class NutrientContent {
  final String nutrient;
  final double content;
  final charts.Color color;

  NutrientContent(this.nutrient, this.content, Color color)
      : this.color = charts.Color(
      r: color.red, g: color.green, b: color.blue, a: color.alpha);
}



