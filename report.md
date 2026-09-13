# AMERICAN SIGN LANGAUGE ALPHABET IDENTIFIER 
[Youtube Link](https://youtu.be/XOHV7dQ894s)

## MOTIVATION
I am ideating to build an end-to-end real-time sign language interpreter which acts as an interpreter in a conversation between a Deaf and hearing individual. I used the capstone as the starting point to help me build this product. I am using a dataset of images of letters in American Sign Language. The aim of this project is to use the MNIST dataset of signs and augment the data to introduce variations and capture signs from different angles. This ensures that signs can be identified from noisy images. I analyse the model performance by using CNN and CNN + XGBoost and report my observations. 

## RUNNING PROJECT 
Running the `starter_notebook.ipynb` suffices because it contains everything inluding installing requirements to loading the dataset (operations performed in `data.py`), creating the 3 models (from model.py), training the model using gradient descent and XGBoost (from train.py) and evaluating the model by calculating the values across multiple seeds, the mean and standard deviation and the confusion matrix. 

## DATASET
### How I choose the dataset? 
0. [Sign Langauge MNIST Dataset](https://www.kaggle.com/datasets/datamunge/sign-language-mnist?resource=download) is the dataset I chose. Metadata about the dataset is available on the hyperlink. 
1. Some datasets in Kaggle weren’t used by others => they're probably not good and it's better to avoid them.  
2. [Roboflow](https://universe.roboflow.com/david-lee-d0rhs/american-sign-language-letters) had a good dataset but MNIST is a standard technique, so I used that instead.  
3. Avoided datasets with signs of words because I think having videos of word signs is better to capture the movement. Hencewhy I liked this MNIST dataset because it excluded j and z (2 letters which require the most movement, g does to some extent too)  

### Augmentation  
1. Augmentation that was performed in dataset to scale from 1704 original color images to 34627 images across train and test set:  
“An image pipeline was used based on ImageMagick and included cropping to hands-only, gray-scaling, resizing, and then creating at least 50+ variations to enlarge the quantity. The modification and expansion strategy was filters ('Mitchell', 'Robidoux', 'Catrom', 'Spline', 'Hermite'), along with 5% random pixelation, +/- 15% brightness/contrast, and finally 3 degrees rotation. Because of the tiny size of the images, these modifications effectively alter the resolution and class separation in interesting, controllable ways." [1](https://www.kaggle.com/datasets/datamunge/sign-language-mnist?resource=download) 

2. The way I implemented augmentation:  
a. I created a flag augment instead of performing augmentation on the entire dataset. I didn't augment test images since the model would be measuring accuracy on augmented images instead of unseen and real images. Similary I didn't augment the validation set since early stopping picks the best epoch on the basis of validation accuracy. This is implemented in `make_loaders(..)`

b. `AugmentingSignsDataset` is a wrapper class which tells the dataloader how to fetch the numpy arrays and handle per-image preprocessing by standardising the dataset. 

c. PSB limiations for augmentations I initially planned on performing

## MODELS USED
### 1. MLP 
Using Lab 4 in which I created a baseline MLP model whose's architecture is flatten -> hidden -> logits.
### 2. CNN 
Aim: extract features form images 
I implemented the CNN's architecture as two convolutional blocks + one FC hidden layer + output layer. 
### 3. CNN + XGBOOST 
Embedding is the output of the FC hidden layer, which is inputted to XGBoost to create the 3rd model CNN + XGBoost.

## EVALUATION 
1. Training Loss of MLP and CNN drops drastically  from epoch 0 to 1 and remains almost stable from epoch 2 onwards. 
2. MLP baseline test acc: 0.7809 +/- 0.0042
per-seed: [0.7752370329057445, 0.7852760736196319, 0.782069157836029]
3. CNN test acc: 0.8979 +/- 0.0081
per-seed: [0.8987730061349694, 0.8876185164528723, 0.9072783045175683]

=> CNN is more accurate than MLP as expected since CNN has more layers and is used for images. 

4. CNN+XGBoost test acc: 0.8921 +/- 0.0042
per-seed: [0.8866424986056888, 0.8966815393195762, 0.8930563301728945]

=> CNN + XGBoost's score is lower than CNN which is unexpected since I added XGBoost to imrpove prediction accuracy. 


## LIMITATIONS
1. I planned on augmenting by applying torchvision.transforms's RandomRotation, RandomAffine (for different perspective), ColorJitter

2. I planned on comparing the augmented dataset's robustness instead of just comparing the CNN and XGBoost architecture. The aim here was to apply  augmentation style changes to the test set and measure robutsness in outlier cases e.g. with poor lighting or resolution of images. I ignored this in the interest of time. 

3. The model accuracy is on the lower end ~0.89 for CNN and CNN + XGBoost. Changing parameters like number of epochs or seeds could help improve accuracy. 

## FUTURE SCOPE (for the MVP and beyond): 
1. End-to-end flow of the MVP of real-time sign language interpreter :  
a. Sign-to-Speech:  
Deaf user’s signs (video) → decode signs and expressions (Computer Vision (CV), Encoder-Decoder architectures (Transformer, LSTM, RNN), CNN) → Sign language grammar --> English grammar → Text-to-speech.  
b. Speech-to-Sign:  
Hearing user’s speech recognition (audio) → Speech-to-text (ASR, NLP) → Sign language grammar (sequence-to-sequence LLM) → expressive avatars which sign hearing user’s speech and interprets by speaking deaf user’s signs (Transformer + GAN)  

2. Simple Improvements: 
a. Enable sign recognition w one hand
b. Fix drawbacks and accessibility issues - like poor camera resolution 

3. A feature to help with the fundamental needs of life like interpreting at a hospital or government services like passport renewal. E.g. At the hospital it helps the Deaf person understand the medical jargons, diagnosis, treatments, recovery, etc.  

4. Incorporating other sign (ISL, General Sign Language) and spoken (Hindi, Bengali) languages and customising the app for each user by identifying unique or culture specific words and sign patterns they use. 

5. Music and movies translation - adding an interpreter signing real time. Includes more emotional aspects since voice intonation of actors or tempo of music expresses different things.  

6. Creating different features of the app for different situations. E.g. Interpreting for a person on a video/ audio call including on video conferencing softwares like Zoom (maybe providing an extension on zoom) 

7. A feature to help Deaf people across different workspaces. Starting point can be common ones like in the service industry (cafe, airport, etc) and expanding to Deaf professionals in business or specific industries with terminologies like tech. Industry specific terminologies will have very different and specific signing styles – almost like a new language. 

## REFERENCES 
1. [https://www.kaggle.com/datasets/datamunge/sign-language-mnist?resource=download](https://www.kaggle.com/datasets/datamunge/sign-language-mnist?resource=download)
