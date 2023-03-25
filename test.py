import numpy as np
import cvxopt
from sklearn.model_selection import train_test_split


class SVM:
    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Convert to cvxopt matrices
        K = np.dot(X, X.T)
        P = cvxopt.matrix(np.outer(y, y) * K)
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        A = cvxopt.matrix(y, (1, n_samples))

        b = cvxopt.matrix(0.0)

        # Inequality constraint (y_i * alpha_i >= 0)
        G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
        h = cvxopt.matrix(np.zeros(n_samples))

        # Equality constraint (sum(alpha_i * y_i) = 0)
        G_std = cvxopt.matrix(np.diag(np.ones(n_samples)))
        h_std = cvxopt.matrix(np.zeros(n_samples))
        G = cvxopt.matrix(np.vstack((G, G_std)))
        h = cvxopt.matrix(np.vstack((h, h_std)))

        # Solve QP problem
        cvxopt.solvers.options['show_progress'] = False
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        # Lagrange multipliers
        multipliers = np.ravel(solution['x'])

        # Support vectors have non zero lagrange multipliers
        self.sv_indices = multipliers > 1e-5
        self.multipliers = multipliers[self.sv_indices]
        self.sv_samples = X[self.sv_indices]
        self.sv_labels = y[self.sv_indices]

        # Bias
        self.bias = 0
        for n in range(len(self.multipliers)):
            self.bias += self.sv_labels[n]
            self.bias -= np.sum(self.multipliers * self.sv_labels * K[n, self.sv_indices])
        self.bias /= len(self.multipliers)

    def predict(self, X):
        result = np.dot(X, self.sv_samples.T) + self.bias
        result = np.sign(result)
        return result

from sklearn.datasets import make_blobs

# Generate a synthetic 2D classification dataset with overlapping classes
X, y = make_blobs(n_samples=500, centers=2, random_state=0, cluster_std=1.0)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Fit the SVM on the training set
svm = SVM()
svm.fit(X_train, y_train)

# Evaluate the SVM on the test set
y_pred = svm.predict(X_test)
accuracy = np.mean(y_pred == y_test)
print("Test set accuracy: {:.2f}".format(accuracy))

import matplotlib.pyplot as plt

def plot_decision_boundary(X, y, svm):
    # Get the min and max values of the features
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    # Create a grid of points to evaluate the classifier on
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    # Flatten the grid of points to a single list of features
    X_grid = np.c_[xx.ravel(), yy.ravel()]

    # Predict the labels for the grid of points
    predictions = svm.predict(X_grid)
    predictions = predictions.reshape(xx.shape)

    # Plot the decision boundary
    plt.contourf(xx, yy, predictions, cmap=plt.cm.coolwarm, alpha=0.8)

    # Plot the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.show()

# Fit the SVM on the training set
svm = SVM()
svm.fit(X_train, y_train)

# Plot the decision boundary
plot_decision_boundary(X_train, y_train, svm)
