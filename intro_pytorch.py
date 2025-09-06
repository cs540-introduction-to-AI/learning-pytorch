import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
def learn_torch():
    data=[[1,2],[3,4]]
    tensor_Data=torch.tensor(data)
    nparray=np.array(data)
    tensor_Data2=torch.from_numpy(nparray)
    torch_ones=torch.ones_like(data)
    print(f"Ones Tensor: \n {torch_ones} \n")
    torch_rand = torch.rand_like(data, dtype=torch.float) # overrides the datatype of x_data
    print(f"Random Tensor: \n {torch_rand} \n")
def get_data_loader(training = True):
    """
    TODO: implement this function.

    INPUT: 
        An optional boolean argument (default value is True for training dataset)

    RETURNS:
        Dataloader for the training set (if training = True) or the test set (if training = False)
    """
    custom_transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    train_set=datasets.FashionMNIST('./data',train=True,
    download=True,transform=custom_transform)
    test_set=datasets.FashionMNIST('./data', train=False,
    transform=custom_transform)
    test_loader= torch.utils.data.DataLoader(test_set, batch_size = 64,shuffle=False)
    train_loader=torch.utils.data.DataLoader(train_set, batch_size = 64)
    if training==True:
        return train_loader
    return test_loader
def build_model():
    """
    TODO: implement this function.

    INPUT: 
        None

    RETURNS:
        An untrained neural network model
    """
    model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28*28, 128),
    nn.ReLU(),
    nn.Linear(128,64),
    nn.ReLU(),
    nn.Linear(64,10))
    return model
def train_model(model, train_loader, criterion, T):
    #  opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    #  model.train()
    #  for epoch in range(T):
    #     total_loss = 0
    #     correct = 0
    #     for data, labels in train_loader:
    #         opt.zero_grad()
    #         output = model(data)
    #         loss = criterion(output, labels)
    #         loss.backward()
    #         opt.step()
    #         total_loss += loss.item() * data.size(0)
    #         pred = output.argmax(dim=1, keepdim=True)
    #         correct += pred.eq(labels.view_as(pred)).sum().item()
    #     print(f'Train Epoch: {epoch} Accuracy: {correct}/{len(train_loader.dataset)}({100. * correct / len(train_loader.dataset):.2f}%) Loss: {total_loss / len(train_loader.dataset):.3f}')
    
     """
    TODO: implement this function.

    INPUT: 
        model - the model produced by the previous function
        train_loader  - the train DataLoader produced by the first function
        criterion   - cross-entropy 
        T - number of epochs for training

    RETURNS:
        None
    """
     
     size_dataset=len(train_loader.dataset)
     criterion = nn.CrossEntropyLoss()
     opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
     for epoch in range(T):
        running_loss=0.0
        model.train()
        count=0
        for feature_vectors,labels in train_loader:
            opt.zero_grad()
            # print(feature_vectors.shape)
            # print(feature_vectors[0])
            outputs=model(feature_vectors)
            size=labels.size()[0]
            for k in range(size):
                max=-100
                store=0
                check=labels[k].item()
                for l in range(10):
                    if outputs[k][l]>max:
                        max=outputs[k][l]
                        store=l
                if check==store:
                    count=count+1
            loss=criterion(outputs,labels)
            loss.backward()
            opt.step()
            running_loss += loss.item()
      
            #accumulated loss will keep on decreasing on every epoch r trial of training of model ono the dataset-for 10 epoches,we train the model 10 times on the dataset
            #and we keep getting the accumulated loss as the most minimum in the all previous epoches
        #print("Train Epoch: {} Accuracy: {}/{}({}%) Loss: {}".format(epoch,a,i,a/i,running_loss/(i)))#alternate way of printing without lessening  number of significant digits
        print(f'Train Epoch: {epoch} Accuracy: {count}/{size_dataset}({100*count/size_dataset:.2f}%) Loss: {running_loss/1000:.3f}')
        
def evaluate_model(model, test_loader, criterion, show_loss = True):
    """
    TODO: implement this function.

    INPUT: 
        model - the the trained model produced by the previous function
        test_loader    - the test DataLoader
        criterion   - cropy-entropy 

    RETURNS:
        None
    """
    size_dataset=len(test_loader.dataset)
    criterion = nn.CrossEntropyLoss()
    opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    #here T=1(only run once on the testing dataset)
    T=1
    for epoch in range(T):
        running_loss=0.0
        model.eval()
        count=0
        for feature_vectors,labels in test_loader:
                opt.zero_grad()
                outputs=model(feature_vectors)
                size=labels.size()[0]
                for k in range(size):
                    max=-100
                    store=0
                    check=labels[k].item()
                    for l in range(10):
                        if outputs[k][l]>max:
                            max=outputs[k][l]
                            store=l
                    if check==store:
                        count=count+1
                loss=criterion(outputs,labels)
                loss.backward()
                opt.step()
                running_loss += loss.item()
            #accumulated loss will keep on decreasing on every epoch r trial of training of model ono the dataset-for 10 epoches,we train the model 10 times on the dataset
            #and we keep getting the accumulated loss as the most minimum in the all previous epoches
        #print("Train Epoch: {} Accuracy: {}/{}({}%) Loss: {}".format(epoch,a,i,a/i,running_loss/(i)))#alternate way of printing without lessening  number of significant digits
        #print(f'Train Epoch: {epoch} Accuracy: {count}/{size_dataset}({100*count/size_dataset:.2f}%) Loss: {running_loss/1000:.3f}')
        if show_loss==True:
            print(f'Average loss: {running_loss/10000:.4f}')
        print(f'Accuracy: {100*count/size_dataset:.2f}%')
def predict_label(model, test_images, index):
    """
    TODO: implement this function.

    INPUT: 
        model - the trained model
        test_images   -  a tensor. test image set of shape Nx1x28x28
        index   -  specific index  i of the image to be tested: 0 <= i <= N - 1


    RETURNS:
        None
    """
    #we are given a feature vector(an image) of 28*28 size 
    #in train_model() method,we had a matrix of feature vectors(64 columns in the matrix for 64 feature vectors/images)
    #here we are given only one image without its corresponding label and we have to figure out the label
    #get only one image as the test dataset
    hardcoded_names=['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle Boot']
    k=0
    for element in test_images:
        if k==index:
            test_images=element
            break
        k=k+1
    #criterion = nn.CrossEntropyLoss()
    opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    #here T=1(only run once on the testing dataset)
    model.eval()
    opt.zero_grad()
    outputs=model(test_images)
    #print(outputs)#has dimensions=1(#of rows)
    prob=F.softmax(outputs,1)#therefore should also be given dimensions=1
    #print(torch.sort(prob,descending=True))
    prob_value,prob_index=torch.sort(prob,descending=True)
    for i in range(3):
        prob1,index=prob_value[0][i],prob_index[0][i]
        print(f'{hardcoded_names[index]}: {prob1*100:.2f}%')
if __name__ == '__main__':
    '''
    Feel free to write your own test code here to exaime the correctness of your functions. 
    Note that this part will not be graded.
    '''
   # criterion = nn.CrossEntropyLoss()
    train_loader = get_data_loader()
    print(type(train_loader))
    print(train_loader.dataset)#gives the number of training items
    print(len(train_loader))#gives number of batches in the dataset
    test_loader = get_data_loader(False)
    model=build_model()
    print(model)
    train_model(model,train_loader,1,5)
    evaluate_model(model,test_loader,1,True)
    k=0
    m=6
    for element,label in test_loader:
        if k>m:
            predict_label(model,element,1)#gets m+1th element of test_loader
            break
        k=k+1
    


