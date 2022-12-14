import numpy as np

class KNN(object):
    """
        kNN classifier object.
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
        #### YOUR CODE HERE! 
        ###
        ##
        
        self.task_kind = 'classification'
        self.set_arguments(*args, **kwargs)

    def set_arguments(self, *args, **kwargs):
        """
            args and kwargs are super easy to use! See dummy_methods.py
            The KNN class should have a variable defining the number of neighbours (k).
            You can either pass this as an arg or a kwarg.
        """
        ##
        ###
        #### YOUR CODE HERE! 
        ###
        ##
        if "k" in kwargs:
            self.k = kwargs["k"]
        elif len(args) > 0:
            self.k = args[0]
        else:
            print("using default values in knn")
            self.k = 100

    

    def fit(self, training_data, training_labels):
        """
            Trains the model, returns predicted labels for training data.
            Hint: Since KNN does not really have parameters to train, you can try saving the training_data
            and training_labels as part of the class. This way, when you call the "predict" function
            with the test_data, you will have already stored the training_data and training_labels
            in the object.
            
            Arguments:
                training_data (np.array): training data of shape (N,D)
                training_labels (np.array): labels of shape (N,)
            Returns:
                pred_labels (np.array): labels of shape (N,)
        """

        ##
        ###
        #### YOUR CODE HERE! 
        ###
        ##

        self.training_data = training_data
        self.traning_labels = training_labels

        return self.predict(training_data)
                               
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
        #### YOUR CODE HERE! 
        ###
        ##
        return test_labels
    
    def normalize(data, means, stds):
        """This function takes the data, the means,
        and the standard deviatons(precomputed). It 
        returns the normalized data.
        
        Inputs:
            data : shape (NxD)
            means: shape (1XD)
            stds : shape (1xD)
            
        Outputs:
            data_normed: shape (NxD)
        """
        # WRITE YOUR CODE HERE
        # return the normalized features
        
        return (data - means) / stds

    def euclidean_dist(example, training_examples):
        """function to compute the Euclidean distance between a single example
        vector and all training_examples

        Inputs:
            example: shape (D,)
            training_examples: shape (NxD) 
        Outputs:
            return distance vector of length N
        """
        
        # WRITE YOUR CODE HERE
        return np.sqrt(np.sum((training_examples - example.T) ** 2, axis = 1))

    def find_k_nearest_neighbors(k, distances):
        """ Find the indices of the k smallest distances from a list of distances.
            Tip: use np.argsort()
        """
        
        # WRITE YOUR CODE HERE
        indices = np.argsort(distances)[:k]
        
        return indices