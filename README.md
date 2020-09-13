# ML Algorithm with Python

## Table of Contents
  1. [Linear Regression](#linear_regression)
  2. [Logistic Regression](#logistic_regression)
  3. [Neural Network](#neural_network)

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
## 2. [Logistic Regression (with Regularization)](/logistic_regression/logistic_regression_reg.ipynb)
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
  <img src="/logistic_regression/images/decision_boundary_overfitting.png" width="250" />
  <img src="/logistic_regression/images/decision_boundary_underfitting.png" width="250" />
  <img src="/logistic_regression/images/decision_boundary_regularization.png" width="250" />
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
...
