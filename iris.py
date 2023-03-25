import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.optimize import linprog

def non_linear_svm(X, y, C=1.0, gamma=1.0):
  n_samples = len(X)
  
  K = np.zeros((n_samples, n_samples))
  for i in range(n_samples):
    for j in range(n_samples):
      K[i, j] = rbf_kernel(X[i], X[j], gamma)
  
  # Create the P and q matrices
  P = np.outer(y, y) * K
  q = -np.ones(n_samples)
  
  # Create the G and h matrices
  G = np.vstack((np.diag(-np.ones(n_samples)), np.identity(n_samples)))
  h = np.hstack((np.zeros(n_samples), np.ones(n_samples) * C))
  
  # Create the A and b matrices
  A = y.reshape(1, -1)
  b = np.zeros(1)
  
  # Solve the quadratic optimization problem
  solution = linprog(q, G, h, A, b)
  
  # Get the Lagrange multipliers
  lagr_mult = np.ravel(solution.x)
  
  # Get the support vectors and their indices
  support_vectors = X[lagr_mult > 1e-7]
  support_vector_indices = np.arange(len(lagr_mult))[lagr_mult > 1e-7]
  
  # Get the non-zero Lagrange multipliers
  non_zero_lagr_mult = lagr_mult[lagr_mult > 1e-7]
  
  b = 0
  for i in range(len(non_zero_lagr_mult)):
    b += y[support_vector_indices[i]]
    b -= np.sum(non_zero_lagr_mult * y[support_vector_indices[i]] * K[support_vector_indices[i], support_vector_indices])
  b /= len(non_zero_lagr_mult)
  
  return support_vectors, non_zero_lagr_mult, b


def rbf_kernel(x1, x2, gamma):
  return np.exp(-gamma * np.linalg.norm(x1 - x2) ** 2)

def predict(X, support_vectors, non_zero_lagr_mult, b, gamma):
  y_pred = []
  for sample in X:
    prediction = 0
    for i in range(len(non_zero_lagr_mult)):
      prediction += non_zero_lagr_mult[i] * y[i] * rbf_kernel(support_vectors[i], sample, gamma)
    prediction += b
    y_pred.append(np.sign(prediction))
  return np.array(y_pred)


# Generate a non-linear dataset
X, y = make_moons(n_samples=500, noise=0.2, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

support_vectors, non_zero_lagr_mult, b = non_linear_svm(X_train, y_train, C=1.0, gamma=1.0)

y_pred = predict(X_test, support_vectors, non_zero_lagr_mult, b, gamma=1.0)

accuracy = np.mean(y_pred == y_test)
print(f"Accuracy: {(accuracy * 100)}%")

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

# Make predictions on the grid of points
# Z = predict(np.c_[xx.ravel(), yy.ravel()], support_vectors, non_zero_lagr_mult, b, gamma=1.0)
# Z = Z.reshape(xx.shape)

# Visualize the decision boundary
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='cool')
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

#? desperate grid trial
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = predict(xy, support_vectors, non_zero_lagr_mult, b, gamma=1.0)


#? msh 3aref a3mekl plot
Z = Z.reshape(XX.shape)
plt.pcolormesh(XX, YY, Z, cmap='cool', alpha=0.1)
ax.scatter(support_vectors[:, 0], support_vectors[:, 1], s=100, linewidth=1, facecolors='none', edgecolors='k')
plt.show()
