package perceptron

class Perceptron {

  private var bias: Double = 0
  private val weights: Array[Double] = Array(0.0, 0.0)

  def fit(x_train: Array[Array[Double]], y_train: Array[Int], learning_rate: Double, max_epochs: Int = 100): Unit = {

    // Loop through each epoch w/ a max of max_epochs
    for (i <- 0 until max_epochs) {
      // Initialize error
      var sum_error = 0.0

      // Loop through each row
      for ((row, idx) <- x_train.zipWithIndex) {
        // call activation function for initial prediction
        val prediction = predict(row)

        // find error of init. prediction
        val error = y_train(idx) - prediction
        sum_error += scala.math.pow(error, 2)

        // update the bias
        bias = bias + learning_rate * error

        // update each weight
        for ((weight, idx) <- weights.zipWithIndex) {
          weights(idx) = weight + learning_rate * error * row(idx)
        }
      }
      println(s"epoch=$i :: learning rate=$learning_rate :: error=$sum_error")
    }

  }

  def predict(row: Array[Double]): Int = {
    // init activation to the bias
    var activation = bias

    // Iterate through values in row and apply update function
    for ((value, idx) <- row.zipWithIndex) {
      activation += value * weights(idx)
    }

    // Apply stepper function
    if (activation >= 0.0) 1 else 0
  }
}
