# Deep Learning for Alzheimer's Disease Risk Detection
A Deep Learning framework for predicting the risk of developing Alzheimer's Dementia.


## 1.Data

The Kaggle dataset of MRI images used as input for this project can be found here: 
https://www.kaggle.com/tourist55/alzheimers-dataset-4-class-of-images
Once you download it, there are 4 points throughout the code, where the file directory needs to be updated based on where you saved the file.
   
 1.<img width="443" alt="Screenshot 2021-08-13 at 8 45 31 PM" src="https://user-images.githubusercontent.com/88878812/129433122-c089f826-1a52-4aaf-845e-5682fb497ea5.png">

 2.<img width="447" alt="Screenshot 2021-08-13 at 8 29 19 PM" src="https://user-images.githubusercontent.com/88878812/129432805-6640325d-bde0-406e-8aa2-b6eccee8a5a8.png">

 3.<img width="447" alt="Screenshot 2021-08-13 at 8 34 57 PM" src="https://user-images.githubusercontent.com/88878812/129432926-c0c49269-fca3-4989-94d3-71fa90e43a2c.png"> 

 **in this one: you have to access one of the 4 final directories in each iterarion, so you have to give the path like this "./Alzheimer_s dataset/train/" (in contrast to: "./Alzheimer_s dataset/train").
 
 4.<img width="447" alt="Screenshot 2021-08-13 at 8 39 59 PM" src="https://user-images.githubusercontent.com/88878812/129433017-4da21e9e-ac8d-4180-bc2e-ef7b70f93214.png">
 
 In the same directory as the .ipynb and .py files, please set your directory up as so:
 
 	Alzheimer_s Dataset/
 
    	train/
	 
        		classes
		  
    	test/
	 
       		classes

## 2.Basic structure
We have created 2 CNNs that output a classification result based on 4 dementia states: Non-Demented, Very Mild Demented, Mild Demented, Moderate Demented.
Our goal was to use 2 different structures of RNN (one custom and one with Transfer Learning on AlexNet) and compare them to see, which one would give us the prediction with the highest accuracy based on the .jpg images from Kaggle Dataset.

## 3.To run the code:
Update the directories and run both nn_alexzheimernet.py and nn_customCNN.ipyb to see the epochs AND the graphs. For just the epochs and no graphs, run the nn.py for the custom CNN and alexzheimers_net.py for the AlexNet model. 

You will also need to have the following packages installed:
- os
- tensorflow
- numpy
- pandas
- matplotlib
- PIL





