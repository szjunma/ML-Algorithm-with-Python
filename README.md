# ML Algorithm with Python

<p float="none">
  <img src="/title.png" width="1000" />
</p>

## Table of Contents
  1. [Linear Regression](#linear_regression)
  2. [Logistic Regression](#logistic_regression)
  3. [Neural Network](#neural_network)
  4. [Decision Tree](#decision_tree)
  5. [K-Means](#k-means)
  6. [Principal Component Analysis](#principal_component_analysis)

## Being constantly updated ..

Machine learning algorithm written in Python
- Requires only basic linear algebra
- Uses Numpy for matrix operation to avoid costly looping in Python
- Usually includes notebook for easy visual
- Simple examples provided

to run each algorithm, `cd` to corresponding directory and run the following (replace `***` with the corresponding algorithm) in terminal:

```
python ***.py
```

(I also have plan to package each algorithm into a class)

## credit
Most equations are from [CS 229](http://cs229.stanford.edu/syllabus-autumn2018.html) class of Stanford.  

<a name="linear_regression"></a>
## 1. [Linear Regression](/linear_regression/linear_regression.ipynb)

### Cost function

```
def cost(X, y, theta):
    h = np.dot(X, theta)
    cos = np.sum((h - y) * ( h - y))/(2 * len(y))
    return cos
```

### Gradient descent
```
def gradient_descent(X, y, theta, alpha, num_iters):
    m = len(y)
    costs = []
    for _ in range(num_iters):
        h = np.dot(X,theta)
        theta -= alpha * np.dot(X.T, (h - y))/m
        costs.append(cost(X, y, theta))
    return theta, costs
```
### Feature normalization
```
def feature_normaliza(X):
    mu = np.mean(X, 0)      
    sigma = np.std(X, 0)   
    def get_norm(col):
        mu = np.mean(col)      
        sigma = np.std(col)
        return (col - mu)/sigma
    return np.apply_along_axis(get_norm, 0, X), mu, sigma
```

### Main function

```
def linear_regression(X, y, alpha = 0.01,num_iters = 100):
    X = np.append(np.ones((X.shape[0], 1)), X, axis = 1)
    theta = np.zeros((X.shape[1], 1), dtype = np.float64)
    theta, costs = gradient_descent(X, y, theta, alpha, num_iters)
    predicted = np.dot(X, theta)
    return predicted, theta, costs
```
### Plot an example
```
predicted, theta, costs = linear_regression(X, y)

plt.plot(X, predicted, 'b')
plt.plot(X, y, 'rx', 10)
for i, x in enumerate(X):
    plt.vlines(x, min(predicted[i], y[i]), max(predicted[i], y[i]))
plt.ylabel('Y')
plt.xlabel('X')
plt.legend(('linear fit', 'data'))
plt.show()
```
<p float="left">
  <img src="/linear_regression/images/Error.png" width="500" />
</p>

<a name="logistic_regression"></a>
## 2. [Logistic Regression (with Regularization)](/logistic_regression/logistic_regression.ipynb)
### Sigmoid function
```
def sigmoid(z):
    return 1/(1 + np.exp(-z))
```

### Cost function

```
def cost_reg(theta, X, y, lam = 0):
    h = sigmoid(np.dot(X, theta))
    theta1 = theta.copy()
    theta1[0] = 0
    cos = -(np.sum(y * np.log(h)) + np.sum((1 - y) * np.log(1 - h)))/len(y) + lam * np.sum(theta1 * theta1)/len(y)
    return cos
```

### Expand features
```
def expand_feature(x1, x2, power = 2):
    #expand a 2D feature matrix to polynimial features up to the power
    new_x = np.ones((x1.shape[0], 1))
    for i in range(1, power + 1):
        for j in range(i + 1):
            new_x = np.append(new_x, (x1**(i-j)*(x2**j)).reshape(-1, 1), axis = 1)
    return new_x
```

### Gradient descent
```
def gradient_descent_reg(X, y, theta, alpha, lam = 0, num_iters = 100):
    m = len(y)
    costs = []

    for _ in range(num_iters):
        h = sigmoid(np.dot(X, theta))
        theta1 = theta.copy()
        theta1[0] = 0
        theta -= alpha * (np.dot(X.T, (h - y)) + 2 * lam * theta1)/m
        costs.append(cost_reg(theta, X, y))
    return theta, costs
```

### Prediction
```
def predict(theta, X):
    return (sigmoid(np.dot(X, theta)) > 0.5).flatten()
```
### Main function
```
def logistic_regression_reg(X, y, power = 2, alpha = 0.01, lam = 0, num_iters = 100):
    X = expand_feature(X[:, 0], X[:, 1], power = power)
    theta = np.zeros((X.shape[1], 1), dtype = np.float64)
    theta, costs = gradient_descent_reg(X, y, theta, alpha, lam, num_iters)
    predicted = predict(theta, X)
    return predicted, theta, costs
```
### Examples
<p float="left">
  <img src="/logistic_regression/images/decision_boundary_overfitting.png" width="300" />
  <img src="/logistic_regression/images/decision_boundary_underfitting.png" width="300" />
  <img src="/logistic_regression/images/decision_boundary_regularization.png" width="300" />
</p>

<a name="neural_network"></a>
## 3. [Neural Network](/neural_network/neural_network.ipynb)
### Initialize parameters
```
def init_para(D, K, h):
    W = np.random.normal(0, 0.01, (D, h))
    b = np.zeros((1, h), dtype = float)
    W2 = np.random.normal(0, 0.01, (h, K))
    b2 = np.zeros((1, K), dtype = float)
    return W, b, W2, b2
```
### Softmax
```
def softmax(scores):
    exp_scores = np.exp(scores)
    return exp_scores / np.sum(exp_scores, axis = 1).reshape(-1, 1)
```

### Main function
```
def nnet(X, y, step_size = 0.4, lam = 0.0001, h = 10, num_iters = 1000):
    # get dim of input
    N, D = X.shape
    K = y.shape[1]

    W, b, W2, b2 = init_para(D, K, h)

    # gradient descent loop to update weight and bias
    for i in range(num_iters):
        # hidden layer, ReLU activation
        hidden_layer = np.maximum(0, np.dot(X, W) + np.repeat(b, N, axis = 0))

        # class score
        scores = np.dot(hidden_layer, W2) + np.repeat(b2, N, axis = 0)

        # compute and normalize class probabilities
        probs = softmax(scores)

        # compute the loss with regularization
        data_loss = np.sum(-np.log(probs) * y) / N
        reg_loss = 0.5 * lam * np.sum(W * W) + 0.5 * lam * np.sum(W2 * W2)
        loss = data_loss + reg_loss

        # check progress
        if i%1000 == 0 or i == num_iters:
            print("iteration {}: loss {}".format(i, loss))

        # compute the gradient on scores
        dscores = (probs - y) / N

        # backpropate the gradient to the parameters
        dW2 = np.dot(hidden_layer.T, dscores)
        db2 = np.sum(dscores, axis = 0)
        # next backprop into hidden layer
        dhidden = np.dot(dscores, W2.T)
        # backprop the ReLU non-linearity
        dhidden[hidden_layer <= 0] = 0
        # finally into W,b
        dW = np.dot(X.T, dhidden)
        db = np.sum(dhidden, axis = 0)

        # add regularization gradient contribution
        dW2 = dW2 + lam * W2
        dW = dW + lam * W

        # update parameter
        W = W - step_size * dW
        b = b - step_size * db
        W2 = W2 - step_size * dW2
        b2 = b2 - step_size * db2
    return W, b, W2, b2
```

### Example
<p float="left">
  <img src="/neural_network/images/decision_boundary_nnet.png" width="500" />
</p>

<a name="decision_tree"></a>
## 4. [Decision Tree](/decision_tree/decision_tree.ipynb)
### Gini impurity/Entropy
```
def gini_impurity(y):
    # calculate gini_impurity given labels/classes of each example
    m = y.shape[0]
    cnts = dict(zip(*np.unique(y, return_counts = True)))
    impurity = 1 - sum((cnt/m)**2 for cnt in cnts.values())
    return impurity

def entropy(y):
    # calculate entropy given labels/classes of each example
    m = y.shape[0]
    cnts = dict(zip(*np.unique(y, return_counts = True)))
    disorder = - sum((cnt/m)*log(cnt/m) for cnt in cnts.values())
    return disorder
```

### Information gain
```
def info_gain(l_y, r_y, cur_gini):
    # calculate the information gain for a certain split
    m, n = l_y.shape[0], r_y.shape[0]
    p = m / (m + n)
    return cur_gini - p * gini_impurity(l_y) - (1 - p) * gini_impurity(r_y)
```

### Find best split
```
def get_split(X, y):
    # loop through features and values to find best combination with the most information gain
    best_gain, best_index, best_value = 0, None, None

    cur_gini = gini_impurity(y)
    n_features = X.shape[1]  

    for index in range(n_features):  

        values = np.unique(X[:, index], return_counts = False)  

        for value in values:  

            left, right = test_split(index, value, X, y)

            if left['y'].shape[0] == 0 or right['y'].shape[0] == 0:
                continue

            gain = info_gain(left['y'], right['y'], cur_gini)

            if gain > best_gain:
                best_gain, best_index, best_value = gain, index, value
    best_split = {'gain': best_gain, 'index': best_index, 'value': best_value}
    return best_split
```
### Create leaf and decision node
```
class Leaf:
    # define a leaf node
    def __init__(self, y):
        self.counts = dict(zip(*np.unique(y, return_counts = True)))
        self.prediction = max(self.counts.keys(), key = lambda x: self.counts[x])

class Decision_Node:
    # define a decision node
    def __init__(self, index, value, left, right):
        self.index, self.value = index, value
        self.left, self.right = left, right
```
### Training (build decision tree)
```
def decision_tree(X, y, max_dep = 5, min_size = 10):
    # train the decision tree model with a dataset
    correct_prediction = 0

    def build_tree(X, y, dep, max_dep = max_dep, min_size = min_size):
        # recursively build the tree
        split = get_split(X, y)

        if split['gain'] == 0 or dep >= max_dep or y.shape[0] <= min_size:
            nonlocal correct_prediction
            leaf = Leaf(y)
            correct_prediction += leaf.counts[leaf.prediction]
            return leaf

        left, right = test_split(split['index'], split['value'], X, y)

        left_node = build_tree(left['X'], left['y'], dep + 1)
        right_node = build_tree(right['X'], right['y'], dep + 1)

        return Decision_Node(split['index'], split['value'], left_node, right_node)

    root = build_tree(X, y, 0)

    return correct_prediction/y.shape[0], root
```
### Prediction
```
def predict(x, node):
    if isinstance(node, Leaf):
        return node.prediction

    if x[node.index] < node.value:
        return predict(x, node.left)
    else:
        return predict(x, node.right)
```


### Example
<p float="left">
  <img src="/decision_tree/images/decision_boundary.png" width="1000" />
</p>

<a name="k-means"></a>
## 5. [K-Means](/k-means/k-means.ipynb)
### Initialize centroids
```
def init_centroid(X, K):
    m = X.shape[0]
    idx = np.random.choice(m, K, replace = False)
    return X[idx, :]
```
### Update labels
```
def update_label(X, centroid):
    m, K = X.shape[0], centroid.shape[0]
    dist = np.zeros((m, K))
    label = np.zeros((m, 1))

    for i in range(m):
        for j in range(K):
            dist[i,j] = np.dot((X[i, :] - centroid[j, :]).T, (X[i, :] - centroid[j, :]))

    label = np.argmin(dist, axis = 1)
    total_dist = np.sum(np.choose(label, dist.T))
    return label, total_dist
```
### Update centroids
```
def update_centroid(X, label, K):
    D = X.shape[1]
    centroid = np.zeros((K, D))
    for i in range(K):
        centroid[i, :] = np.mean(X[label.flatten() == i, :], axis=0).reshape(1,-1)
    return centroid
```
### K-Means function
```
def k_means(X, K, num_iters = 100):
    m = X.shape[0]
    centroid = init_centroid(X, K)

    for _ in range(num_iters):
        label, total_dist = update_label(X, centroid)
        centroid = update_centroid(X, label, K)

    return centroid, label, total_dist
```
### Example
<p float="left">
  <img src="/k-means/images/k-means.png" width="500" />
</p>

### Determine K
<p float="left">
  <img src="/k-means/images/total_dist_vs_k.png" width="500" />
</p>

<a name="principal_component_analysis"></a>
## 6. [Principal Component Analysis](/PCA/PCA.ipynb)

### SVD (Singular Value Decomposition)
```
Sig = np.dot(X_norm.T,X_norm)/X_norm.shape[0]
U,S,V = np.linalg.svd(Sig)
```

### Data projection
```
def project_data(X_norm, U, K):
    Z = np.zeros((X_norm.shape[0], K))
    U_reduce = U[:, 0:K]
    Z = np.dot(X_norm, U_reduce)
    return Z
```

### Data recovery
```
def recover_data(Z, U, K):
    X_rec = np.zeros((Z.shape[0], U.shape[0]))
    U_recude = U[:, 0:K]
    X_rec = np.dot(Z, U_recude.T)
    return X_rec
```

### PCA function
```
def PCA(X, K):
    X_norm, mu, sigma = feature_normaliza(X)

    Sig = np.dot(X_norm.T,X_norm)/X_norm.shape[0]
    U,S,V = np.linalg.svd(Sig)

    Z = project_data(X_norm, U, K)
    X_rec = recover_data(Z, U, K)
    return X_rec
```
### Example (2D -> 1D)
<p float="left">
  <img src="/PCA/images/PCA.png" width="500" />
</p>

...
