import perceptron.Perceptron

object Main extends App {
  val x_train = Array(Array(2.7810836,2.550537003),
    Array(1.465489372,2.362125076),
    Array(3.396561688,4.400293529),
    Array(1.38807019,1.850220317),
    Array(3.06407232,3.005305973),
    Array(7.627531214,2.759262235),
    Array(5.332441248,2.088626775),
    Array(6.922596716,1.77106367),
    Array(8.675418651,-0.242068655),
    Array(7.673756466,3.508563011))

  val y_train = Array(0, 0, 0, 0, 0, 1, 1, 1, 1, 1)

  val p = new Perceptron()

  p.fit(x_train, y_train, 0.001)

  for ((row, idx) <- x_train.zipWithIndex) {
    val prediction = p.predict(row)
    val actual = y_train(idx)
    val x0 = row(0)
    val x1 = row(1)
    println(f"$x0 $x1 pred=$prediction actual=$actual")
  }
}

