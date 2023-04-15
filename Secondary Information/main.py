# this is the main f training loop file 
import utils
import torch
import torchvision
from torchvision import datasets, transforms
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler


if __name__ == '__main__':
    dataset = utils.MyDataset1('ptg1.xlsx')
    # Calculate the sizes of the training and validation sets
    # remove the effeect of having classes baised distribution 
    # split the data set 
    train_size = int(0.7 * len(dataset))
    val_size = len(dataset) - train_size

    # Split the dataset into training and validation sets
    batch_size=300
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create data loaders for the training and validation sets
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    net = utils.Net()
    #net=utils.ResidualNet()
    #class_weights = torch.tensor([(1-0.8063), (1-0.0784), (1-0.1153)])
    #loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    lambda_l2 = 0.01
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.005, weight_decay=0.001)
    global_it = 0 # defined here, use in your summaries
    #writer=SummaryWriter() # setup the writer directory

    # ==================================================================
    # Train loop
    # ==================================================================
    for epoch in range(100):
        for x, y in train_dataloader:
            # you may need to add steps for normilization.
            # normlize the input data set 
            #mean = torch.mean(x, dim=0)
            #std = torch.std(x, dim=0)
            #x = (x - mean) / std
            optimizer.zero_grad() ## This is was not added at to make the gradiant zero again, after adding it the accurcy was imrproved

            pred = net(x)
            #print(pred)
            y_onehot = F.one_hot(y, num_classes=3).float() # convert to one-hot vector
            #print(y_onehot)
            #loss = loss_fn(pred,y_onehot)
            # add loss with regulaizer
            loss = loss_fn(pred,y_onehot) + lambda_l2 * sum((param**2).sum() for param in net.parameters())
            accuracy = utils.compute_accuracy(pred.cpu().detach().numpy(),y.cpu().detach().numpy())
            loss.backward()
            optimizer.step()

            # print out stats every 10 its
            if global_it % 10 == 0:
                print(f'Iteration: {global_it} | Loss: {loss.item()} | Accuracy: {accuracy}')
            ## store the loss and accuracy values
            #writer.add_scalar("Loss/train", loss,global_it) # add a value of the loss for the writer
            #writer.add_scalar("Accuracy/train", accuracy,global_it) # add a value of the loss for the writer
            # increment
            global_it += 1

    #writer.flush()
    print('Training complete')
    #writer.close()
#%% 
# now create the validation loop of the model
# Validation loop
with torch.no_grad(): # Turn off gradient calculation to speed up computation and reduce memory usage
    net.eval() # Set network to evaluation mode
    total_loss = 0
    total_accuracy = 0
    total_examples = 0
    for x_val, y_val in val_dataloader:
        #mean = torch.mean(x, dim=0)
        #std = torch.std(x, dim=0)
        #x = (x - mean) / std

        pred_val = net(x_val)
        print(pred_val)
        
        y_val_onehot = F.one_hot(y_val, num_classes=3).float() # convert to one-hot vector
        
        loss_val = loss_fn(pred_val, y_val_onehot)
        accuracy_val = utils.compute_accuracy(pred_val.cpu().detach().numpy(),y_val.cpu().detach().numpy())
        total_loss += loss_val.item() * x_val.size(0) # Multiply the loss by the batch size to account for varying batch sizes
        total_accuracy += accuracy_val * x_val.size(0)
        total_examples += x_val.size(0)

    avg_loss = total_loss / total_examples
    avg_accuracy = total_accuracy / total_examples
    print(f'Validation | Loss: {avg_loss} | Accuracy: {avg_accuracy}')