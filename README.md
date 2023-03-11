# PlantInfectionDetector
A Convolutional Neural Network model built to predict the diseases that occur in plants from image data to facilitate early treatment.


What it does?

The project builds a CNN model that aims to classify the diseases occuring in potato plants based on the training datasetc containing images of earlyblight, lateblight and healthy potato plant leaves. 

The CNN model built consists of 5 hidden layers activated using the relu function. The output layer classifies the input into one among the three classes using the softmax function:
 -> Healthy
 -> Early blight
 -> Late blight

An accuracy of around 99% was achieved after 50 epochs

How to test the model?

1. Download the dataset from https://www.kaggle.com/datasets/arjuntejaswi/plant-village into your local
2. Run the training.py module to train the CNN model that also performs testing on a test dataset
3. The model is saved to your local 
