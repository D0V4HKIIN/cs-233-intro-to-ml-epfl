import numpy as np
from metrics import accuracy_fn, mse_fn, macrof1_fn


def splitting_fn(data, labels, indices, fold_size, fold):
    """
        Function to split the data into training and validation folds.
        Arguments:
            data (np.array, of shape (N, D)): data (which will be split to training 
                and validation data during cross validation),
            labels  (np.array, of shape (N,)): the labels of the data
            indices: (np.array, of shape (N,)): array of pre shuffled indices (integers ranging from 0 to N)
            fold_size (int): the size of each fold
            fold (int): the index of the current fold.
        Returns:
            train_data, train_label, val_data, val_label (np. arrays): split training and validation sets
    """

    ##
    ###
    # YOUR CODE HERE!
    ###
    ##

    # validation indicies of length fold_size
    # min is to not go outside of the array
    val_indices = indices[(fold * fold_size)
                           : min(((fold + 1) * fold_size), data.shape[0] - 1)]
    # train indicies. The rest
    train_indices = np.setdiff1d(indices, val_indices)

    train_data = data[train_indices]
    train_label = labels[train_indices]
    val_data = data[val_indices]
    val_label = labels[val_indices]

    return train_data, train_label, val_data, val_label


def cross_validation(method_obj=None, search_arg_name=None, search_arg_vals=[], data=None, labels=None, k_fold=4):
    """
        Function to run cross validation on a specified method, across specified arguments.
        Arguments:
            method_obj (object): A classifier or regressor object, such as KNN. Needs to have
                the functions: set_arguments, fit, predict.
            search_arg_name (str): the argument we are trying to find the optimal value for
                for example, for DummyClassifier, this is "dummy_arg".
            search_arg_vals (list): the different argument values to try, in a list.
                example: for the "DummyClassifier", the search_arg_name is "dummy_arg"
                and the values we try could be [1,2,3]
            data (np.array, of shape (N, D)): data (which will be split to training 
                and validation data during cross validation),
            labels  (np.array, of shape (N,)): the labels of the data
            k_fold (int): number of folds
        Returns:
            best_hyperparam (float): best hyper-parameter value, as found by cross-validation
            best_acc (float): best metric, reached using best_hyperparam
    """
    # choose the metric and operation to find best params based on the metric depending upon the
    # kind of task.
    metric = mse_fn if method_obj.task_kind == 'regression' else macrof1_fn
    find_param_ops = np.argmin if method_obj.task_kind == 'regression' else np.argmax

    N = data.shape[0]
    indices = np.arange(N)
    np.random.shuffle(indices)
    fold_size = N//k_fold

    # mean accuracy for every input value
    acc_list1 = []
    for arg in search_arg_vals:
        arg_dict = {search_arg_name: arg}
        # this is just a way of giving an argument
        # (example: for DummyClassifier, this is "dummy_arg":1)
        method_obj.set_arguments(**arg_dict)

        # accuracy for every validation fold
        acc_list2 = []
        for fold in range(k_fold):
            ##
            ###
            # YOUR CODE HERE!
            ###
            ##
            train_data, train_label, val_data, val_label = splitting_fn(
                data, labels, indices, fold_size, fold)
            method_obj.fit(train_data, train_label)
            predicted_data = method_obj.predict(val_data)
            acc_list2.append(metric(predicted_data, val_label))

        ##
        ###
        # YOUR CODE HERE!
        ###
        ##
        acc_list1.append(np.mean(acc_list2))

    ##
    ###
    # YOUR CODE HERE!
    ###
    ##
    best_param = find_param_ops(acc_list1)
    best_hyperparam = search_arg_vals[best_param]
    best_acc = acc_list1[best_param]

    print(acc_list1)

    return best_hyperparam, best_acc
