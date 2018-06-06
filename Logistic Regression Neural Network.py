import numpy as np
#Trying to create a class for logistic regression like one in sk-learn to get an idea of 
#inner workings of the code and how the dataframe/dimensions are handled. Derivatives and
#Cost updates.

class LogisticRegression(object):
    def __init__(self, W=None, b=None):
        self.W = W
        self.b = b

    def sigmoid(self, x):
        return ((1+np.exp(-x))**(-1))

    def initialize_with_zeros(self, dim):
        self.W = np.zeros((dim, 1))
        self.b = 0
        return self.W, self.b

    def propogate(self, X, y):
        m = X.shape[1]
        A = self.sigmoid((np.dot(self.W.T, X)+self.b))
        cost = (-1/m)*(np.dot(y, np.log(A).T)+np.dot(1-y, np.log(1-A).T))
        dW = 1/m * np.dot(X, (A-y).T)
        db = 1/m * np.sum(A-y)
        cost = np.squeeze(cost)
        grads = {'dW': dW, 'db': db}
        return grads, cost

    def optimize(self, X, y, num_iterations, learning_rate, print_cost=False):
        costs = []
        for i in range(num_iterations):
            grads, cost = self.propogate(X, y)
            dW = grads['dW']
            db = grads['db']
            self.W -= learning_rate*dW
            self.b -= learning_rate*db
            if i % 100 == 0:
                costs.append(cost)
            if print_cost and i % 100 == 0:
                print('Cost after {} iterations is {}'.format(i, cost))
        params = {'W': self.W, 'b': self.b}
        grads = {'dW': dW, 'db': db}
        return params, grads, costs

    def predict(self, X):
        m = X.shape[1]
        Y_predicted = np.zeros((1, m))
        self.W = self.W.reshape(X.shape[0], 1)
        A = self.sigmoid(np.dot(self.W.T, X)+self.b)
        for i in range(A.shape[1]):
            Y_predicted = 1 if A[0, i] > 0.5 else 0
        return Y_predicted

    def fit(self, X_train, y_train, X_test, y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
        self.W, self.b = self.initialize_with_zeros(X_train.shape[0])
        parameters, grads, costs = self.optimize(
            X_train, y_train, num_iterations, learning_rate, print_cost)
        self.W, self.b = parameters['W'], parameters['b']
        y_predicted_test = self.predict(X_test)
        y_predicted_train = self.predict(X_train)
        print('Train accuracy: {}'.format(
            100-np.mean(np.abs(y_predicted_train-y_train))*100))
        print('Test accuracy: {}'.format(
            100-np.mean(np.abs(y_predicted_test-y_test))*100))
        d = {"costs": costs,
             "Y_prediction_test": y_predicted_test,
             "Y_prediction_train": y_predicted_train,
             "w": self.W,
             "b": self.b,
             "learning_rate": learning_rate,
             "num_iterations": num_iterations}
        return d

model = LogisticRegression()