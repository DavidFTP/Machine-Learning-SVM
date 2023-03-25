from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def svm(X, y, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
    n_features = X.shape[1]

    # Convert y to an array where samples with y <= 0 are assigned -1, and samples with y > 0 are assigned 1
    y_ = np.where(y <= 0, -1, 1)

    w = np.zeros(n_features)
    b = 4

    for _ in range(n_iters):
        for idx, x_i in enumerate(X):
            # Check if the sample is correctly classified
            condition = y_[idx] * (np.dot(x_i, w) - b) >= 1
            if condition:
                # If the sample is correctly classified, update the weight vector and bias term with the regularization term
                w -= learning_rate * (2 * lambda_param * w)
            else:
                # If the sample is misclassified, update the weight vector and bias term with the regularization term and the sample's contribution
                w -= learning_rate * (2 * lambda_param * w - np.dot(x_i, y_[idx]))
                b -= learning_rate * y_[idx]
    return w, b

def predict(X, w, b):
    approx = np.dot(X, w) - b
    # returning the sign because its a binary representation
    return np.sign(approx)

def accuracy(y_true, y_pred):
    accuracy = (np.sum(y_true == y_pred) / len(y_true)) * 100
    return f'{accuracy}%'

def decisionBoundary(x, w, b, offset):
    return (-w[0] * x + b + offset) / w[1]


#! Separable Dataset
iris = sns.load_dataset('iris')
# basta5dem 2 species bas that are linearly separable 
iris = iris[iris.species.isin(['setosa', 'virginica'])]
# esta5demt el petal length wel width as the feature el hyeb2a el x wel y axis when plotting
X = iris[["petal_length", "petal_width"]].values
# Redefine the output data (species)
# y = iris["species"].replace({"setosa": -1, "virginica": 1}).values
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.1, random_state=42)
#! Non-separable Dataset
# Generate a 2D classification dataset with overlapping classes
X, y = make_blobs(n_samples=500, centers=2, random_state=0, cluster_std=1.0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



#! bedayet el support vector machine
w, b = svm(X_train, y_train)
predictions = predict(X_test, w, b)

print(f'SVM classification accuracy {accuracy(y_test, predictions)}')

#! plotting everything
def visualize_svm():
    x0_1 = np.amin(X[:, 0])
    x0_2 = np.amax(X[:, 0])

    x1_1 = decisionBoundary(x0_1, w, b, 0)
    x1_2 = decisionBoundary(x0_2, w, b, 0)
    plt.plot([x0_1, x0_2], [x1_1, x1_2], "y--")

    # Decision boundary for the separable dataset
    # sns.scatterplot(x='petal_length', y='petal_width', hue='species', data=iris)
    # Decision boundary for the non-separable dataset
    sns.scatterplot(x=X[:,0], y=X[:,1], hue=y, data=X)
    plt.show()

visualize_svm()