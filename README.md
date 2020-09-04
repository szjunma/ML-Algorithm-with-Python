## ML Algorithm with Python

### Being constantly updated ...

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

### credit
Most equations are from [CS 229](http://cs229.stanford.edu/syllabus-autumn2018.html) class of Stanford.  

### 1. Linear regression

#### cost function

```
def cost(X, y, theta):
    h = np.dot(X, theta)
    cos = np.sum((h - y) * ( h - y))/(2 * len(y))
    return cos
```

#### gradient descent
```
def gradient_descent(X, y, theta, alpha, num_iters):
    m = len(y)
    costs = []
    for _ in range(num_iters):
        h = np.dot(X,theta)
        theta -= alpha * (np.sum((h - y) * X, axis = 0)).reshape(X.shape[1], 1)/m
        costs.append(cost(X, y, theta))
    return theta, costs
```
#### feature normalization
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

#### main function

```
def linear_regression(X, y, alpha = 0.01,num_iters = 100):
    X = np.append(np.ones((X.shape[0], 1)), X, axis = 1)
    theta = np.zeros((X.shape[1], 1), dtype = np.float64)
    theta, costs = gradient_descent(X, y, theta, alpha, num_iters)
    predicted = np.dot(X, theta)
    return predicted, theta, costs
```

#### plot an example
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

### 2. Logistic regression




...
