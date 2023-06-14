## FastApi project that contains endpoint with convolutional neural network that can predict class of an uploaded image.

### About the model
The model is a simple convolutional neural network with SE (squeeze and excitation) blocks, trained on the CIFAR-10 dataset. 


### Dependencies:

1. Clone the project's source code from the repository:
```git clone https://github.com/Arencius/Cifar10-classifier-FastAPI.git```
2. Navigate to the project's directory:
```cd Cifar10-classifier-FastAPI```
3. Install all necessary dependecies:
```pip install -r requirements.txt```

### Usage:
In order to start the server, run this command in your terminal:
```uvicorn main:app --reload```. Open your web browser and navigate to http://localhost:8000/docs to view the endpoint. There, you can upload an image, and the trained neural network will try to predict the class it belongs to.
