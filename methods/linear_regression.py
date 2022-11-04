from xml.etree.ElementPath import prepare_predicate
import numpy as np
import sys


class LinearRegression(object):
    """
        Linear regressor object. 
        Note: This class will implement BOTH linear regression and ridge regression.
        Recall that linear regression is just ridge regression with lambda=0.
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
        # YOUR CODE HERE! DONE
        ###
        ##
        self.task_kind = 'regression'
        self.set_arguments(*args, **kwargs)

    def set_arguments(self, *args, **kwargs):
        """
            args and kwargs are super easy to use! See dummy_methods.py
            In case of ridge regression, you need to define lambda regularizer(lmda).

            You can either pass these as args or kwargs.
        """

        ##
        ###
        # YOUR CODE HERE! DONE
        ###
        ##

        if "lmda" in kwargs:
            self.lmda = kwargs["lmda"]
        # if not, then check if args is a list with size bigger than 0.
        elif len(args) > 0:
            self.lmda = args[0]
        # if there were no args or kwargs passed, we set the lmda to 1 (default value).
        else:
            print("using default values in linear regression")
            self.lmda = 0

    def fit(self, training_data, training_labels):
        """
            Trains the model, returns predicted labels for training data.
            Arguments:
                training_data (np.array): training data of shape (N,D)
                training_labels (np.array): regression target of shape (N,regression_target_size)
            Returns:
                pred_regression_targets (np.array): predicted target of shape (N,regression_target_size)
        """

        ##
        ###
        # YOUR CODE HERE! DONE
        ###
        ##

        # random ridge stuff idk
        self.w = np.linalg.inv(training_data.T @ training_data +
                               self.lmda * np.identity(training_data.shape[1])) @ training_data.T @ training_labels

        return self.predict(training_data)

    def predict(self, test_data):
        """
            Runs prediction on the test data.

            Arguments:
                test_data (np.array): test data of shape (N,D)
            Returns:
                pred_regression_targets (np.array): predicted targets of shape (N,regression_target_size)
        """

        ##
        ###
        # YOUR CODE HERE! DONE
        ###
        ##

        return test_data @ self.w
