from utils import label_to_onehot
import numpy as np
import sys
sys.path.append('..')


class LogisticRegression(object):
    """
        LogisticRegression classifier object.
        Feel free to add more functions to this class if you need.
        But make sure that __init__, set_arguments, fit and predict work correctly.
    """

    def __init__(self, *args, **kwargs):
        """
            Initialize the task_kind (see dummy_methods.py)
            and call set_arguments function of this class.
        """

        ##
        ###
        # YOUR CODE HERE!
        ###
        ##
        self.task_kind = 'classification'
        self.set_arguments(*args, **kwargs)

    def set_arguments(self, *args, **kwargs):
        """
            args and kwargs are super easy to use! See dummy_methods.py
            The LogisticRegression class should have variables defining the learning rate (lr)
            and the number of max iterations (max_iters)
            You can either pass these as args or kwargs.
        """

        ##
        ###
        # YOUR CODE HERE!
        ###
        ##

        # first checks if "lr" or "max_iters" was passed as a kwarg.
        if "lr" in kwargs:
            self.lr = kwargs["lr"]
        if "max_iters" in kwargs:
            self.max_iters = kwargs["max_iters"]
        # if not, then check if args is a list with size bigger than 1.
        elif len(args) > 1:
            self.lr = args[0]
            self.max_iters = args[1]
        # if there were no args or kwargs passed, we set the lr and max_iters to 0.03 and 500 respectively (default value).
        else:
            print("using default values in logistic regression")
            self.lr = 1e-3
            self.max_iters = 500

    def fit(self, training_data, training_labels):
        """
            Trains the model, returns predicted labels for training data.
            Arguments:
                training_data (np.array): training data of shape (N,D)
                training_labels (np.array): regression target of shape (N,)
            Returns:
                pred_labels (np.array): target of shape (N,)
        """

        ##
        ###
        # YOUR CODE HERE!
        ###
        ##

        num_classes = training_labels.reshape(
            training_labels.shape[0], -1).shape[1]

        self.isBinary = (num_classes == 1)

        if self.isBinary:
            self.w = np.random.normal(0., 0.1, [training_data.shape[1]])
            self.w = self.logistic_regression_train(
                training_data, training_labels)
        else:
            self.w = np.random.normal(
                0., 0.1, (training_data.shape[1], num_classes))
            self.w = self.logistic_regression_train_multi(
                training_data, training_labels)

        return self.predict(training_data)

    def logistic_regression_train(self, data, labels):

        for it in range(self.max_iters):

            gradient = self.gradient_logistic(data, labels)
            self.w = self.w - self.lr * gradient

            predictions = self.logistic_regression_classify(data)
            if self.accuracy_fn(labels, predictions) == 1:
                break

        return self.w

    def logistic_regression_train_multi(self, data, labels):

        for it in range(self.max_iters):

            gradient = self.gradient_logistic_multi(data, labels)
            self.w = self.w - self.lr * gradient

            predictions = self.logistic_regression_classify_multi(data)
            if self.accuracy_fn(np.argmax(labels, axis=1), predictions) == 1:
                break

        return self.w

    def gradient_logistic(self, data, labels):
        return data.T @ (self.sigmoid(data @ self.w) - labels)

    def gradient_logistic_multi(self, data, labels):

        grad_w = data.T @ (self.f_softmax(data) - labels)
        return grad_w

    def sigmoid(self, t):
        divider = 1 + np.exp(-t)
        return 1/divider

    def f_softmax(self, data):

        res = np.zeros((data.shape[0], self.w.shape[1]))

        for j in range(self.w.shape[1]):
            res[::, j] = (np.exp(data @ self.w[::, j]) /
                          np.sum(np.exp(data @ self.w), axis=1)).T

        return res

    def logistic_regression_classify(self, data):

        predictions = self.sigmoid(data @ self.w)
        predictions[predictions < 0.5] = 0
        predictions[predictions >= 0.5] = 1
        return predictions

    def logistic_regression_classify_multi(self, data):

        predictions = self.f_softmax(data)
        predictions = np.argmax(predictions, axis=1)

        return predictions

    def accuracy_fn(self, labels_gt, labels_pred):

        acc = np.sum(labels_gt == labels_pred) / labels_gt.shape[0]
        return acc

    def predict(self, test_data):
        """
            Runs prediction on the test data.

            Arguments:
                test_data (np.array): test data of shape (N,D)
            Returns:
                test_labels (np.array): labels of shape (N,)
        """
        ##
        ###
        # YOUR CODE HERE! DONE :)
        ###
        ##

        if self.isBinary:
            return self.logistic_regression_classify(test_data)
        else:
            return self.logistic_regression_classify_multi(test_data)
