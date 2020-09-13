# ML Algorithm with Python

## Being constantly updated ..

Machine learning algorithm written in Python
- Requires basic linear algebra
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

## 1. Linear regression

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
![error in linear regression](/linear_regression/images/Error.png)



## 2. Logistic regression (with regularization)
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
  <img src="/logistic_regression/images/decision_boundary_overfitting.png" width="200" />
  <img src="/logistic_regression/images/decision_boundary_underfitting.png" width="200" />
  <img src="/logistic_regression/images/decision_boundary_regularization.png" width="200" />
</p>
...
