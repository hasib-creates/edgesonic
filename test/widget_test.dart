// This is a basic Flutter widget test.
//
// To perform an interaction with a widget in your test, use the WidgetTester
// utility in the flutter_test package. For example, you can send tap and scroll
// gestures. You can also use WidgetTester to find child widgets in the widget
// tree, read text, and verify that the values of widget properties are correct.

import 'package:edge_sonic_app/main.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  testWidgets('EdgeSonic home screen renders anomaly detection UI', (tester) async {
    await tester.pumpWidget(const EdgeSonicApp());

    // Allow asynchronous model loading attempt to complete.
    await tester.pumpAndSettle();

    expect(find.text('EdgeSonic Anomaly Detection'), findsAtLeastNWidgets(1));
    expect(find.text('Model Not Loaded'), findsOneWidget);
    expect(find.text('Select Audio File'), findsOneWidget);
  });
}
