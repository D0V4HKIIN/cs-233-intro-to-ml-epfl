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
        self.task_kind = 'regression'
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
                training_labels (np.array): regression target of shape (N,regression_target_size)
            Returns:
                pred_labels (np.array): target of shape (N,regression_target_size)
        """

        ##
        ###
        # YOUR CODE HERE!
        ###
        ##
        self.w = np.random.normal(0., 0.1, [training_data.shape[1], ])
        for it in range(self.max_iters):
            # write your code here: find gradient and do a gradient step
            gradient = self.gradient_logistic(
                training_data, training_labels)
            self.w = self.w - self.lr * gradient
            ##################################

            # if we reach 100% accuracy, we can stop training immediately
            # predictions = self.predict(training_data, self.w)
            # if accuracy_fn(training_labels, predictions) == 1:
            #     break

        return self.predict(training_data)

    def gradient_logistic(self, data, labels):
        return data.T @ (self.sigmoid(data @ self.w) - labels)

    def sigmoid(self, t):
        divider = 1 + np.exp(-t)
        return 1/divider

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

        return test_data @ self.w
