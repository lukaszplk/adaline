import numpy as np


class Perceptron():
    
    def __init__(self, eta=0.1, epochs=50, is_verbose=False):

        self.eta = eta
        self.epochs = epochs
        self.is_verbose=is_verbose
        self.list_of_errors=[] # mozna wyswietlic spadek bledu na wykresie

    def predict(self, X):
        # przyjmuje zagniezdzona liste
        
        ones = np.ones((X.shape[0], 1))
        X_1 = np.append(X.copy(), ones, axis=1)
        return np.where(self.get_activation(X_1)>0, 1, -1)

    def get_activation(self, X):

        activation = np.dot(X, self.w)
        return activation

    def fit(self, X, y):

        self.list_of_errors=[]

        X_1 = np.append(X.copy(),
                        np.ones((X.shape[0], 1)),
                        axis=1)

        self.w = np.random.rand(X_1.shape[1])

        for e in range(self.epochs):

            error = 0

            activation = self.get_activation(X_1)
            delta_w = self.eta * np.dot((y - activation), X_1)
            self.w += delta_w

            error = np.square(y - activation).sum()/2.0

            self.list_of_errors.append(error) 

            if(self.is_verbose):
                print('Epoch: {}, weights: {}, error: {}'.format(e+1, self.w, error))



