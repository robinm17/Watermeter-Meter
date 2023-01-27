import numpy as np
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim
from PIL import Image


# transform function to reformat normal image to tensor format
transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])

# resize function to resize image to 28x28 pixels    
resize = transforms.Resize((28,28))

# function to turn rgb channgel image into grayscale format [3,28,28] -> [1,28,28]
gray = transforms.Grayscale()

# function to convert image gray colors
invert = transforms.RandomInvert()

# values for the AI model
input_size = 784
hidden_sizes = [128, 64]
output_size = 10


imageList = []
string =[]

# initialize AI model
model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[1], output_size),
                      nn.LogSoftmax(dim=1))

# load the trained model to the newly made model
model.load_state_dict(torch.load('_model_'))



# this function shows the plottes predicition with the given image
def view_classify(img, ps):
    ''' Function for viewing an image and it's predicted classes.
    '''
    ps = ps.data.numpy().squeeze()

    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
    ax1.imshow(img.numpy().squeeze())
    ax1.axis('off')
    ax2.barh(np.arange(10), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    ax2.set_yticklabels(np.arange(10))
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)
    plt.tight_layout()
    

# function to receive the images
def receiveImages():
    # im = Image.open("pic2.png")
    image = Image.open("digit_0.jpg")
    image1= Image.open("digit_1.jpg")
    image2= Image.open("digit_2.jpg")
    image3= Image.open("digit_3.jpg")
    image4= Image.open("digit_4.jpg")

    # imageList.append(im)

    imageList.append(image)
    imageList.append(image1)
    imageList.append(image2)
    imageList.append(image3)
    imageList.append(image4)


# function to convert given images to the right format
def convertList(images):
    global imageList
    imageList = []
    for i in images:
        # img = invert(gray(resize(transform(i))))
        # img = invert(resize(transform(i)))

        img = transform(i)
        img = resize(img)
        img = gray(img)
        img = invert(img)
        print(img.shape)
        # print(img)
        # plt.imshow(img.permute(1,2,0))
        # plt.show()
        # img = resize(transform(i))
        imageList.append(img)

# function which makes predictions based on the given image
def predict(img):
    img = img.view(1,784)
    with torch.no_grad():
        logps = model(img)
    # print(logps)

    # # Output of the network are log-probabilities, need to take exponential for probabilities
    ps = torch.exp(logps)
    # print(ps)
    probab = list(ps.numpy()[0])
    # print(probab)
    print("Predicted Digit =", probab.index(max(probab)))

    view_classify(img.view(28,28), ps)

    plt.show()

    return probab.index(max(probab))


# function to re-list predictions into an array
def reassign(list):
    for i in list:
        value = predict(i)
        string.append(value)


receiveImages()    
convertList(imageList)
# for i in imageList:
#     plt.show()

reassign(imageList)