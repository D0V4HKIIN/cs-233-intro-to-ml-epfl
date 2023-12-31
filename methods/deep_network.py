import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from metrics import accuracy_fn, macrof1_fn


## MS2!!


class SimpleNetwork(nn.Module):
    """
    A network which does classification!
    """
    def __init__(self, input_size, num_classes, hidden_size=32):
        super(SimpleNetwork, self).__init__()

        ##
        ###
        #### YOUR CODE HERE! 
        ###
        ##


        # smaller_layer = int(input_size / 20)
        # small_hidden = int(hidden_size / 2)

        # print(input_size, smaller_layer, hidden_size * 2, hidden_size, small_hidden, num_classes)

        print(input_size, hidden_size * 4, hidden_size, num_classes)
        self.fc1 = nn.Linear(input_size, hidden_size * 4)
        self.fc2 = nn.Linear(hidden_size * 4, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        
        # self.fc1 = nn.Linear(input_size, smaller_layer)
        # self.fc2 = nn.Linear(smaller_layer, hidden_size * 2)
        # self.fc3 = nn.Linear(hidden_size * 2, hidden_size)
        # self.fc4 = nn.Linear(hidden_size, small_hidden)
        # self.fc5 = nn.Linear(small_hidden, num_classes)

    def forward(self, x):
        """
        Takes as input the data x and outputs the 
        classification outputs.
        Args: 
            x (torch.tensor): shape (N, D)
        Returns:
            output_class (torch.tensor): shape (N, C) (logits)
        """

        ##
        ###
        #### YOUR CODE HERE! 
        ###
        ##
        

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        # x = F.relu(self.fc4(x))
        # x = self.fc5(x)

        return x

class Trainer(object):

    """
        Trainer class for the deep network.
    """

    def __init__(self, model, lr, epochs, beta=100):
        """
        """
        self.lr = lr
        self.epochs = epochs
        self.model= model
        self.beta = beta

        self.classification_criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)

    def train_all(self, dataloader_train, dataloader_val):
        """
        Method to iterate over the epochs. In each epoch, it should call the functions
        "train_one_epoch" (using dataloader_train) and "eval" (using dataloader_val).
        """
        for ep in range(self.epochs):
            self.train_one_epoch(dataloader_train)
            self.eval(dataloader_val)

            if (ep+1) % 50 == 0:
                print("Reduce Learning rate")
                for g in self.optimizer.param_groups:
                    g["lr"] = g["lr"]*0.8


    def train_one_epoch(self, dataloader):
        """
        Method to train for ONE epoch.
        Should loop over the batches in the dataloader. (Recall the exercise session!)
        Don't forget to set your model to training mode!
        i.e. self.model.train()
        """
        
        ##
        ###
        #### YOUR CODE HERE! 
        ###
        ##
         
        # Training.
        self.model.train()
        
        for it, batch in enumerate(dataloader):
            # 5.1 Load a batch, break it down in images and targets.
            x, _, y = batch

             # 5.2 Run forward pass.
            logits = self.model.forward(x)  # YOUR CODE HERE
            
            # 5.3 Compute loss (using 'criterion').
            loss = self.classification_criterion(logits, y)  # YOUR CODE HERE
            
            # 5.4 Run backward pass.
            loss.backward()  # YOUR CODE HERE
            
            # 5.5 Update the weights using optimizer.
            self.optimizer.step()  # YOUR CODE HERE
            
            # 5.6 Zero-out the accumulated gradients.
            self.optimizer.zero_grad()  # YOUR CODE HERE
        

    def eval(self, dataloader):
        """
            Method to evaluate model using the validation dataset OR the test dataset.
            Don't forget to set your model to eval mode!
            i.e. self.model.eval()

            Returns:
                Note: N is the amount of validation/test data. 
                We return one torch tensor which we will use to save our results (for the competition!)
                results_class (torch.tensor): classification results of shape (N,)
        """
        ##
        ###
        #### YOUR CODE HERE! 
        ###
        ##

        self.model.eval()

        with torch.no_grad():
            predictions = torch.Tensor()
            acc_run_fn = 0
            acc_run_f1 = 0
            for batch in dataloader:
                # Get batch of data.
                x, _, y = batch
                x_predictions = torch.argmax(F.softmax(self.model(x), dim = 1), axis = 1)
                predictions = torch.cat((predictions, x_predictions))
                curr_bs = x.shape[0]
                # print(x_predictions.numpy().shape, y.numpy().shape)
                acc_run_fn += accuracy_fn(x_predictions.numpy(), y.numpy()) * curr_bs
                acc_run_f1 += macrof1_fn(x_predictions.numpy(), y.numpy()) * curr_bs
            
            acc_fn = acc_run_fn / len(dataloader.dataset)
            acc_f1 = acc_run_f1 / len(dataloader.dataset)

        print("Accuracy fn: " + str(acc_fn))
        print("Macro f1: " + str(acc_f1))

        return predictions