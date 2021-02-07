import torch
import torchvision
from torchvision import transforms
from torchvision.transforms import ToTensor
from PIL import Image
from os import listdir
import random
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import random
import argparse
import numpy as np
from scipy import misc
from PIL import Image
import glob
import imageio
import os

import cv2

import matplotlib.pyplot as plt


class SegNet(nn.Module):
    """neural network architecture inspired by SegNet"""

    def __init__(self):
        super(SegNet, self).__init__()
        
        #Encoder
        self.conv1 = nn.Conv2d(3, 64, (3,3), padding=1)
        self.conv2 = nn.Conv2d(64, 64, (3,3), padding=1)
        self.enc1_bn = nn.BatchNorm2d(64)
        self.maxpool1 = nn.MaxPool2d(2,2)

        self.conv3 = nn.Conv2d(64, 128, (3,3), padding=1)
        self.conv4 = nn.Conv2d(128, 128, (3,3), padding=1)
        self.enc2_bn = nn.BatchNorm2d(128)
        self.maxpool2 = nn.MaxPool2d((2,2),2)

        self.conv5 = nn.Conv2d(128, 256, (3,3), padding=1)
        self.conv6 = nn.Conv2d(256, 256, (3,3), padding=1)
        self.conv7 = nn.Conv2d(256, 256, (3,3), padding=1)
        self.enc3_bn = nn.BatchNorm2d(256)
        self.maxpool3 = nn.MaxPool2d((2,2),2)

        self.conv8 = nn.Conv2d(256, 512, (3,3), padding=1)
        self.conv9 = nn.Conv2d(512, 512, (3,3), padding=1)
        self.conv10 = nn.Conv2d(512, 512, (3,3), padding=1)
        self.enc4_bn = nn.BatchNorm2d(512)
        self.maxpool4 = nn.MaxPool2d((2,2),2)

        self.conv11 = nn.Conv2d(512, 512, (3,3), padding=1)
        self.conv12 = nn.Conv2d(512, 512, (3,3), padding=1)
        self.conv13 = nn.Conv2d(512, 512, (3,3), padding=1)
        self.enc5_bn = nn.BatchNorm2d(512)
        self.maxpool5 = nn.MaxPool2d((2,2),2)

        #Decoder
        self.upsample1 = nn.Upsample(scale_factor=2)
        self.conv14 = nn.Conv2d(512,512, (3,3), padding=1)
        self.conv15 = nn.Conv2d(512,512, (3,3), padding=1)
        self.conv16 = nn.Conv2d(512,512, (3,3), padding=1)
        self.dec1_bn = nn.BatchNorm2d(512)

        self.upsample2 = nn.Upsample(scale_factor=2)
        self.conv17 = nn.Conv2d(512,512, (3,3), padding=1)
        self.conv18 = nn.Conv2d(512,512, (3,3), padding=1)
        self.conv19 = nn.Conv2d(512,256, (3,3), padding=1)
        self.dec2_bn = nn.BatchNorm2d(256)

        self.upsample3 = nn.Upsample(scale_factor=2)
        self.conv20 = nn.Conv2d(256,256, (3,3), padding=1)
        self.conv21 = nn.Conv2d(256,256, (3,3), padding=1)
        self.conv22 = nn.Conv2d(256,128, (3,3), padding=1)
        self.dec3_bn = nn.BatchNorm2d(128)

        self.upsample4 = nn.Upsample(scale_factor=2)
        self.conv23 = nn.Conv2d(128,128, (3,3), padding=1)
        self.conv24 = nn.Conv2d(128,64, (3,3), padding=1)
        self.dec4_bn = nn.BatchNorm2d(64)

        self.upsample5 = nn.Upsample(scale_factor=2)
        self.conv25 = nn.Conv2d(64,64, (3,3), padding=1)
        self.conv26 = nn.Conv2d(64,5, (3,3), padding=1)

        self.softmax = nn.Softmax(dim=1)
        

    def forward(self, x):
        #Encoder
        x = F.relu(self.enc1_bn(self.conv2(F.relu(self.conv1(x)))))
        #print(x.size())
        x = self.maxpool1(x)
        #print(x.size())

        x = F.relu(self.enc2_bn(self.conv4(F.relu(self.conv3(x)))))
        #print(x.size())
        x = self.maxpool2(x)
        #print(x.size())
        
        x = F.relu(self.enc3_bn(self.conv7(F.relu(self.conv6(F.relu(self.conv5(x)))))))
        #print(x.size())
        x = self.maxpool3(x)
        #print(x.size())

        x = F.relu(self.enc4_bn(self.conv10(F.relu(self.conv9(F.relu(self.conv8(x)))))))
        #print(x.size())
        x = self.maxpool4(x)
        #print(x.size())

        x = F.relu(self.enc5_bn(self.conv13(F.relu(self.conv12(F.relu(self.conv11(x)))))))
        #print(x.size())
        x = self.maxpool5(x)
        #print(x.size())

        #print()
        #Decoder
        x = F.relu(self.dec1_bn(self.conv16(F.relu(self.conv15(F.relu(self.conv14(self.upsample1(x))))))))
        #print(x.size())
        x = F.relu(self.dec2_bn(self.conv19(F.relu(self.conv18(F.relu(self.conv17(self.upsample2(x))))))))
        #print(x.size())
        x = F.relu(self.dec3_bn(self.conv22(F.relu(self.conv21(F.relu(self.conv20(self.upsample3(x))))))))
        #print(x.size())
        x = F.relu(self.dec4_bn(self.conv24(F.relu(self.conv23(self.upsample4(x))))))
        #print(x.size())
        x = self.conv26(F.relu(self.conv25(self.upsample4(x))))
        #print(x.size())
        
        return x


def create_data(data_start, data_size, batch_size, input_path, target_path, target_dict, real_sequence, is_train):
    """create data for training/validation from img and xml to tensor"""

    transform = transforms.Compose([transforms.Resize((320, 576)),
                                    transforms.ToTensor()])

    input_list = []
    target_list = []
    data = []

    weights = [0, 0, 0, 0, 0]  # weights for cross entropy loss

    pixel_class = []  # single pixel class

    inputs = os.listdir(input_path)
    inputs.sort()
    print("inputs", len(inputs))

    targets = os.listdir(target_path)
    targets.sort()
    print("targets", len(targets))


    for x in range(data_start, data_size):

        if (len(real_sequence) == 0):
            break

        # print("len sequence",len(real_sequence))

        index = random.choice(real_sequence)
        real_sequence.remove(index)

        print(x)

        # if(len(data) == 8 and not is_train):
        #    break

        # if(len(data) == 4):
        #    break

        input = Image.open(input_path + inputs[index])
        input_list.append(transform(input))
        # input_list.append(ToTensor()(input))

        target = Image.open(target_path + targets[index])
        target_tensor = torch.round(transform(target))
        # target_tensor = torch.round(ToTensor()(target))

        if (is_train):
            target_tensor_final = torch.zeros(320, 576, dtype=torch.long)  # cross entropy loss allowed only torch.long
        else:
            target_tensor_final = torch.zeros(5, 320, 576, dtype=torch.long)

        for i in range(320):
            for j in range(576):
                pixel_class = target_dict[tuple(target_tensor[:, i, j].tolist())]

                # print("pixel class", pixel_class)
                # print("tensor", torch.tensor(pixel_class, dtype=torch.long))
                # print("target size", target_tensor_final.size())

                if (is_train):
                    weights[pixel_class] += 1
                    target_tensor_final[i, j] = torch.tensor(pixel_class, dtype=torch.long)
                else:
                    target_tensor_final[:, i, j] = torch.tensor(pixel_class, dtype=torch.long)
                    weights[pixel_class.index(1)] += 1

        target_list.append(target_tensor_final)

        if len(input_list) >= batch_size:
            data.append((torch.stack(input_list), torch.stack(target_list)))

            input_list = []
            target_list = []

            print('Loaded batch ', len(data), 'of ', int(len(inputs) / batch_size))
            print('Percentage Done: ',
                  100 * (len(data) / int(len(inputs) / batch_size)), '%')

    weights = torch.tensor(weights, dtype=torch.float64)
    # weights = 1/(weights/weights.min()) #press weights in [0,1], with maximum value for each class
    return data, weights


def train(train_data, model, optimizer, criterion, device):
    """
    Trains/updates the model for one epoch on the training dataset.

    Parameters:
        train_data (torch tensor): trainset
        model (torch.nn.module): Model to be trained
        optimizer (torch.optim.optimizer): optimizer instance like SGD or Adam
        criterion (torch.nn.modules.loss): loss function like CrossEntropyLoss
        device (string): cuda or cpu
    """

    # switch to train mode
    model.train()

    # iterate through the dataset loader
    i = 0
    losses = []
    for (inp, target) in train_data:
        # transfer inputs and targets to the GPU (if it is available)
        inp = inp.to(device)
        target = target.to(device)

        # compute output, i.e. the model forward
        output = model(inp)

        # calculate the loss
        loss = criterion(output, target)
        # print("loss", loss)
        losses.append(loss)

        print("loss {:.2}".format(loss))
        # compute gradient and do the SGD step
        # we reset the optimizer with zero_grad to "flush" former gradients
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_loss = torch.mean(torch.stack(losses)).item()
    print("avg.loss {:.2}".format(avg_loss))
    return avg_loss

def calc_accuracy(output, target):
    """calculate accuracy from tensor(b,c,x,y) for every category c"""
    accs = []
    acc_tensor = (output == target).int()
    for c in range(target.size(1)):
        correct_num = acc_tensor[:,c].sum().item() #item convert tensor in integer
        #print(correct_num)
        total_num = acc_tensor[:,c].numel()
        #print(total_num)
        accs.append(correct_num/total_num)
    return accs


def calc_precision(output, target):
    """calculate precision from tensor(b,c,x,y) for every category c"""

    precs = []
    for c in range(target.size(1)):
        true_positives = ((output[:, c] - (output[:, c] != 1).int()) == target[:, c]).int().sum().item()
        # print(true_positives)
        false_positives = ((output[:, c] - (output[:, c] != 1).int()) == (target[:, c] != 1).int()).int().sum().item()
        # print(false_positives)

        if (true_positives == 0):
            precs.append(1.0)
        else:
            precs.append(true_positives / (true_positives + false_positives))

    return precs


def calc_recall(output, target):
    """calculate recall from tensor(b,c,x,y) for every category c"""

    recs = []
    for c in range(target.size(1)):
        relevants = (target[:, c] == 1).int().sum().item()
        # print(relevants)
        true_positives = ((output[:, c] - (output[:, c] != 1).int()) == target[:, c]).int().sum().item()
        # print(true_positives)

        if (relevants == 0):
            recs.append(1.0)
        else:
            recs.append(true_positives / relevants)

    return recs

def convert_to_one_hot(tensor, device):
    """converts a tensor from size (b,c,x,y) to (b,c,x,y) one hot tensor for c categorys"""

    for i in range(tensor.size(0)):
        max_idx = torch.argmax(tensor[i], 0, keepdim=True)
        one_hot = torch.FloatTensor(tensor[i].shape).to(device)
        one_hot.zero_()
        tensor[i] = one_hot.scatter_(0, max_idx, 1)


def validate(val_dataset, model, device, categories):
    """
    validate the model with some validationfunctions on the test/validation dataset.

    Parameters:
        val_data (torch tensor): test/validation dataset
        model (torch.nn.module): Model to be trained
        loss (torch.nn.modules.loss): loss function like CrossEntropyLoss
        device (string): cuda or cpu
        categories (list): names of categories
    """
    model.eval()

    # avoid computation of gradients and necessary storing of intermediate layer activations
    with torch.no_grad():

        accs_avg = [0, 0, 0, 0, 0]
        precs_avg = [0, 0, 0, 0, 0]
        recs_avg = [0, 0, 0, 0, 0]
        counter = 0

        for (inp, target) in val_dataset:
            # transfer to device
            inp = inp.to(device)
            target = target.to(device)

            # compute output
            output = model(inp)

            # print("before extra softmax")
            # print(output[:,:,100,100])

            output = model.softmax(output)
            # print("after extra softmax")
            # print(output[:,:,100,100])

            # convert from probabilities to one hot vectors
            convert_to_one_hot(output, device)

            # print("after convert to one hot")
            # print(output[:,:,100,100])

            accs = calc_accuracy(output, target)
            precs = calc_precision(output, target)
            recs = calc_recall(output, target)

            # print("loss {:.2} IOU {:.2}".format(loss,iou))

            for i in range(len(categories)):
                print("category {:10} accuracy {:.2} precision {:.2} recall {:.2} ".format(categories[i], accs[i],
                                                                                           precs[i], recs[i]))
                accs_avg[i] += accs[i]
                precs_avg[i] += precs[i]
                recs_avg[i] += recs[i]

            print()
            counter += 1

    for i in range(len(categories)):
        accs_avg[i] /= counter
        precs_avg[i] /= counter
        recs_avg[i] /= counter

        print("avg.category {:10} accuracy {:.2} precision {:.2} recall {:.2} ".format(categories[i], accs_avg[i],
                                                                                       precs_avg[i], recs_avg[i]))

    return [accs_avg, precs_avg, recs_avg]

def create_rgb_output(data, model, device, dict_reverse):
    """create rgb pictures from model output for data (rgb-image) on device
       parameter:
            data: torch.tensor (b,3,x,y)
            model: torch#######################################################################

    """
    output = model(data.to(device))
    final_output = model.softmax(output)
    convert_to_one_hot(final_output, device)

    real_output_tensor = torch.zeros(data.size(0),3,data.size(2), data.size(3), dtype=torch.float64)

    for x in range(data.size(0)):
        for i in range(data.size(2)):
            for j in range(data.size(3)):
                real_output_tensor[x][:,i,j] = torch.tensor(dict_reverse[tuple(final_output[x,:,i,j].tolist())])

    return real_output_tensor

def plot_tensor(tensor):
    """plot tensor(3,x,y) as rgb-image"""

    plt.imshow(tensor.permute(1,2,0))

def main():
    parser = argparse.ArgumentParser(description='Subway Station Segmentation')
    parser.add_argument('--input', help='path to img input data directory')
    parser.add_argument('--target', help='path to img target data directory')
    parser.add_argument('--content', help='path where the train/validation tensors, model_weights, losses, validation will be saved',
	                    type=str, default='/') 
    parser.add_argument('--train_tensor_size', help='number of images per training_tensor (should be: train_tensor_size % batch_size = 0)', type=int)
    parser.add_argument('--val_tensor_size', help='number of images per training_tensor (should be: train_tensor_size % batch_size = 0)', type=int)
    parser.add_argument('--num_train_tensors', help='number of train tensors (should be: train_tensor_size * num_train_tensors + val_tensor_size = |images|)', type=int)
    parser.add_argument('--model_weights', help='path where your model weights will be loaded, if not defined new weights will initialized', type=str, default='')
    parser.add_argument('--epochs', help='number of training epochs, default: 50', type=int, default=50)
    parser.add_argument('--batch_size', help='batch size for training, default: 8', type=int, default=8)
    parser.add_argument('--learn_rate', help='learning rate for training, default: 0.0001', type=float, default=0.0001)
    parser.add_argument('--momentum', help='momentum for stochastic gradient descent, default: 0.9', type=float, default = 0.9)
    parser.add_argument('--save_cycle', help='save model, loss, validation every save_cycle epochs, default: 5', type=int, default = 5)
    parser.add_argument('--weight_decay', help='weight_decay for stochastic gradient descent, default: 4e5', type=int, default = 4e5)

    args = parser.parse_args()

    input_path = args.input            
    print("input_path", input_path)

    target_path = args.target 
    print("target_path", target_path)

    content_path = args.content #path where the train/validation tensors, model_weights, losses, validation will be saved
    print("content_path", content_path)

    batch_size = args.train_tensor_size
    print("batch_size", batch_size)

    train_tensor_size = args.num_train_tensors #number of images per training_tensor (should be: train_tensor_size % batch_size = 0)
    print("train_tensor_size", train_tensor_size)

    val_tensor_size = args.val_tensor_size #number of images per training_tensor (should be: train_tensor_size % batch_size = 0)
    print("val_tensor_size", val_tensor_size)

    num_train_tensors = args.num_train_tensors #number of train tensors (should be: train_tensor_size * num_train_tensors + val_tensor_size = |images|) 
    print("num_train_tensors", num_train_tensors)

    model_weights_path = args.model_weights #path where your model weights will be loaded
    print("model_weights_path", model_weights_path)

    load_model = False if(model_weights_path == '') else True #if True model will be loaded from model_weights_path
    print("load_model", load_model)

    learn_rate = args.learn_rate #learning rate for training
    print("learn_rate", learn_rate)

    sgd_momentum = args.momentum #momentum for stochastic gradient descent
    print("sgd_momentum", sgd_momentum)

    sgd_weight_decay = args.weight_decay #weight_decay for stochastic gradient descent
    print("sgd_weight_decay", sgd_weight_decay)

    total_epochs = args.epochs #number of training epochs
    print("total_epochs", total_epochs)

    save_cycle = args.save_cycle #save model, loss, validation every save_cycle epochs
    print("save_cycle", save_cycle)


    categories = ["white", "black", "green", "red", "yellow"]

    #for creating rgb pixel to class category (one_hot)
    dict_val = {(0.0, 0.0, 0.0): (0.0, 1.0, 0.0, 0.0, 0.0), #black
                (0.0, 0.0, 1.0): (0.0, 1.0, 0.0, 0.0, 0.0), #black (fail)
                (0.0, 1.0, 0.0): (0.0, 0.0, 1.0, 0.0, 0.0), #green
                (0.0, 1.0, 1.0): (1.0, 0.0, 0.0, 0.0, 0.0), #white (fail)
                (1.0, 0.0, 0.0): (0.0, 0.0, 0.0, 1.0, 0.0), #red
                (1.0, 0.0, 1.0): (1.0, 0.0, 0.0, 0.0, 0.0), #white (fail)
                (1.0, 1.0, 0.0): (0.0, 0.0, 0.0, 0.0, 1.0), #yellow
                (1.0, 1.0, 1.0): (1.0, 0.0, 0.0, 0.0, 0.0)} #white

    #for making model output to real output
    dict_reverse = {(0.0, 1.0, 0.0, 0.0, 0.0) : (0.0, 0.0, 0.0), #black
                    (0.0, 0.0, 1.0, 0.0, 0.0) : (0.0, 1.0, 0.0), #green
                    (0.0, 0.0, 0.0, 1.0, 0.0) : (1.0, 0.0, 0.0), #red
                    (0.0, 0.0, 0.0, 0.0, 1.0) : (1.0, 1.0, 0.0), #yellow
                    (1.0, 0.0, 0.0, 0.0, 0.0) : (1.0, 1.0, 1.0)} #white

    #for creating rgb pixel to class category (single value, cross entropyloss only allows single value)
    dict_train = {(0.0, 0.0, 0.0): 1, #black
                  (0.0, 0.0, 1.0): 1, #black (fail)
                  (0.0, 1.0, 0.0): 2, #green
                  (0.0, 1.0, 1.0): 0, #white (fail)
                  (1.0, 0.0, 0.0): 3, #red
                  (1.0, 0.0, 1.0): 0, #white (fail)
                  (1.0, 1.0, 0.0): 4, #yellow
                  (1.0, 1.0, 1.0): 0} #white

    real_sequence = list(range(len(os.listdir(input_path)))) #create a list from [0,...,number of input pictures-1] !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    indices = [i*train_tensor_size for i in range(num_train_tensors +1)] #size of train tensors always has to be rejusted !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    for i in range(1,len(indices)): 
        train_data, weights = create_data(indices[i-1],indices[i],batch_size, input_path, target_path, dict_train, real_sequence, True)
        torch.save(train_data, content_path + "Train_Tensor" + str(i) + ".pt")
        torch.save(weights, content_path + "Train_Weights" + str(i) + ".pt")

    real_sequence = list(range(len(os.listdir(input_path))))
    val_data, _ = create_data(0,val_tensor_size, batch_size, input_path, target_path, dict_val, real_sequence, False) #always has to be rejusted
    torch.save(val_data, content_path + "Val_Tensor_Test.pt")


    # set a boolean flag that indicates whether a cuda capable GPU is available
    # we will need this for transferring our tensors to the device and
    # for persistent memory in the data loader
    is_gpu = torch.cuda.is_available()
    print("GPU is available:", is_gpu)
    print("If you are receiving False, try setting your runtime to GPU")

    # set the device to cuda if a GPU is available
    device = torch.device("cuda" if is_gpu else "cpu")

    #create model
    model = SegNet().to(device)

    if(load_model):
        model.load_state_dict(torch.load(model_weights_path))#####################################################################

    #define loss function

    weights = torch.load(content_path + "/Train_Weights_Test1.pt")

    for i in range(2, num_train_tensors):
        weights += torch.load(content_path + "/drive/My Drive/Train_Weights" + str(i) + ".pt")

    weights = 1/(weights/weights.min()) #press weights in [0,1], with maximum value for each class
    weights = weights.type(torch.FloatTensor)
    weights = weights.to(device)
    print("weights", weights)

    criterion = nn.CrossEntropyLoss(weights)

    #set optimizer for backpropagation
    optimizer = torch.optim.SGD(model.parameters(), lr= learn_rate, momentum = sgd_momentum, weight_decay= sgd_weight_decay)

    print(model)

    val_list = []
    loss_list = []

    val_data = torch.load(content + "Val_Tensor.pt")
    for epoch in range(0, total_epochs):

        print("EPOCH:", epoch + 1)
        print("TRAIN")
        for i in range(1, num_train_tensors +1): #tensor_number):
            print("train_data_number:", i+1)
            train_data = torch.load(content_path + "Train_Tensor" +str(i) +".pt")
            loss_list.append(train(train_data, model, optimizer, criterion, device))
        print("VALIDATION")
        val_list.append(validate(val_data, model, device, categories))
    
        if ((epoch) % save_cycle == 0):
            torch.save(model.state_dict(), content_path + "Model_weights_" + str(epoch) + ".pt")
            torch.save(val_list, content_path + "val_list.pt")
            torch.save(loss_list, content_path + "loss_list.pt")

if __name__ == "__main__":
    main()
