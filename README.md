# Watermeter-Meter

This is a project to recognize the digits shown on a watermeter. We are using an ESP32-CAM with an selfmade casing mount to take picture of a watermeter. At the moment the AI is not able to detect the place of digits, for this we are using software to cut out the digits and feed it to the network. 

We are using pytorch as our platform combined with PIL for image editing.

requirements.txt contains all necessery libs to run the code.

### training the model:

Code for training the model can be found in: [**train.py**](train.py)

##### downloading the dataset:
For the model training we use the **MNIST dataset** which contains images of handwritten digits with their corresponding labels
![download](https://user-images.githubusercontent.com/43373858/216311685-33c820b6-037d-4f39-819b-3bf5e22693f9.png)

During downloading the data is transformed and saved in **PATH_TO_STORE_TRAINSET** and **PATH_TO_STORE_TESTSET**.

We transform every image with an size of 28x28 pixels to a Tensor: 

`transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])`

##### the model:
For our network model we use 784 input nodes (28x28 = 784) with 2 hidden layers and 1 output layer:![model](https://user-images.githubusercontent.com/43373858/216312830-c3a0f4e8-ad72-4d97-bba1-0ae8ec272eb9.png)

The model can easily be made using PyTorch's torch.nn module.
