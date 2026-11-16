import unittest
import numpy as np
from grader_contracts.linear_models import RegressionInput, ClassificationInput
from linear_models import train_linear_regression, train_logistic_classifier


class TestLinearRegression(unittest.TestCase):
    def test_weights_convergence(self):
        np.random.seed(0)
        X = np.random.randn(100, 2)
        y = X.dot(np.array([3.0, -2.0])) + 5.0
        
        data = RegressionInput(
            train_features=X[:80], train_target=y[:80],
            test_features=X[80:], test_target=y[80:],
            learning_rate=0.1, epochs=1000
        )
        res = train_linear_regression(data)
        
        np.testing.assert_allclose(res.weights, [3.0, -2.0], rtol=0.05)
        self.assertAlmostEqual(res.bias, 5.0, delta=0.1)


class TestLogisticClassifier(unittest.TestCase):
    def test_accuracy_on_separable_data(self):
        np.random.seed(1)
        X = np.random.randn(200, 2)
        y = (X[:, 0] - X[:, 1] > 0).astype(float)
        
        data = ClassificationInput(
            train_features=X[:150], train_target=y[:150],
            test_features=X[150:], test_target=y[150:],
            learning_rate=0.1, epochs=1000
        )
        res = train_logistic_classifier(data)
        
        acc = np.mean(res.predictions == y[150:])
        self.assertGreater(acc, 0.9)
        
        self.assertLess(res.loss_history[-1], res.loss_history[0])


if __name__ == "__main__":
    unittest.main()