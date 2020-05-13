import random

class MyStochasticGDRegressor:
    def __init__(self):
        self.intercept_ = 0.0
        self.coef_ = []

    # simple stochastic GD
    def fit(self, inputs, outputs, learningRate = 0.0001, noEpochs = 2500):
        inputs=inputs.values.tolist()
        outputs=outputs.values.tolist()
        self.coef_ = [0.0 for _ in range(len(inputs[0]) + 1)]    #beta or w coefficients y = w0 + w1 * x1 + w2 * x2 + ...
        for epoch in range(noEpochs):
            for i in range(len(inputs)): # for each sample from the training data
                ycomputed = self.eval(inputs[i])     # estimate the output
                crtError = float(ycomputed) - float(outputs[i])     # compute the error for the current sample
                for j in range(0, len(inputs[0])):   # update the coefficients
                    self.coef_[j] = self.coef_[j] - learningRate * crtError * float(inputs[i][j])
                self.coef_[len(inputs[0])] = self.coef_[len(inputs[0])] - learningRate * crtError * 1

        self.intercept_ = self.coef_[-1]
        self.coef_ = self.coef_[:-1]

    def eval(self, xi):
        yi = self.coef_[-1]
        for j in range(len(xi)):
            yi += float(self.coef_[j]) * float(xi[j])
        return yi

    def predict(self, x):
        x=x.values.tolist()
        yComputed = [self.eval(xi) for xi in x]
        return yComputed


class MyBatchGDRegressor:
    def __init__(self):
        self.intercept_ = 0.0
        self.coef_ = []


    def fit(self, inputs, outputs, learningRate=0.005, noEpochs=1000):
        inputs = inputs.values.tolist()
        outputs = outputs.values.tolist()
        if not isinstance(inputs[0],list):
            inputs=[[el] for el in inputs]
        self.coef_ = [0.0 for _ in range(len(inputs[0]) + 1)]  # beta or w coefficients y = w0 + w1 * x1 + w2 * x2 + ...

        for epoch in range(noEpochs):
            mean=0
            for i in range(len(inputs)):  # for each sample from the training data
                ycomputed = self.eval(inputs[i])  # estimate the output
                crtError = float(ycomputed) - float(outputs[i])  # compute the error for the current sample
                #print("Err: "+str(crtError))
                mean+=crtError
            mean/=len(inputs)
            #print("Error mean: "+str(mean))
            for i in range(len(inputs)):
                for j in range(0, len(inputs[0])):  # update the coefficients
                    self.coef_[j] = self.coef_[j] - learningRate * mean #* float(inputs[i][j])
                self.coef_[len(inputs[0])] = self.coef_[len(inputs[0])] - learningRate * mean * 1

        self.intercept_ = self.coef_[-1]
        self.coef_ = self.coef_[:-1]


    def eval(self, xi):
        yi = self.coef_[-1]
        lenn=0
        if not isinstance(xi, list):
            return yi + self.coef_[0]*float(xi)

        for j in range(len(xi)):
           yi += float(self.coef_[j]) * float(xi[j])
        return yi


    def predict(self, x):
        x = x.values.tolist()
        yComputed = [self.eval(xi) for xi in x]
        return yComputed


class MyBatchNonLinearGDRegressor:
    def __init__(self):
        self.intercept_ = 0.0
        self.coef_ = []


    def fit(self, inputs, outputs, learningRate=0.005, noEpochs=1000):
        inputs = inputs.values.tolist()
        outputs = outputs.values.tolist()
        self.coef_ = [0.0 for _ in range(len(inputs[0]) + 1)]  # beta or w coefficients y = w0 + w1 * x1 + w2 * x2 + ...

        for epoch in range(noEpochs):
            mean=0
            for i in range(len(inputs)):  # for each sample from the training data
                ycomputed = self.eval(inputs[i])  # estimate the output
                crtError = float(ycomputed) - float(outputs[i])  # compute the error for the current sample
                mean+=crtError
            mean/=len(inputs)
            for i in range(len(inputs)):
                for j in range(0, len(inputs[0])):  # update the coefficients
                    self.coef_[j] = self.coef_[j] - learningRate * mean #* float(inputs[i][j])
                self.coef_[len(inputs[0])] = self.coef_[len(inputs[0])] - learningRate * mean * 1

        self.intercept_ = self.coef_[-1]
        self.coef_ = self.coef_[:-1]


    def eval(self, xi):
        yi = self.coef_[-1]
        for j in range(len(xi)):
           yi += float(self.coef_[j]) * (float(xi[j])**2)  #b0 + b1 * x1^2 + b2 * x2^2
        return yi


    def predict(self, x):
        x = x.values.tolist()
        yComputed = [self.eval(xi) for xi in x]
        return yComputed

class SKGDRegressor:
    def __init__(self):
        self.intercept_ = 0.0
        self.coef_ = []
        self.regr=None

    def fit(self, inputs, outputs, learningRate=0.005, noEpochs=1000):
        from sklearn import linear_model
        self.regr=linear_model.SGDRegressor(max_iter=noEpochs,tol=learningRate)
        self.regr.fit(inputs,outputs)
        self.intercept_,self.coef_=self.regr.intercept_,self.regr.coef_


    def predict(self, x):
        x = x.values.tolist()
        return self.regr.predict(x)