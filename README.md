# Deepfake Detection
<img width="994" height="317" alt="image" src="https://github.com/user-attachments/assets/660ab211-33ba-4bf9-ac1c-858a4d43733f" />

## 📌 Overview
This project focuses on detecting deepfakes in images and videos using state-of-the-art deep learning models and physiological signal analysis. 
With the rapid rise of hyper-realistic manipulated media, detecting such fakes has become crucial to counter misinformation, defamation, and digital security threats.

**Our approach integrates:**
- Image-based detection using CNN architectures.

- Video-based detection using remote Photoplethysmography (rPPG) for physiological signal inconsistencies.

- A hybrid approach combining both methods for improved generalization.


## 🚨 Problem Statement
With the rapid advancement of AI tools, generating hyper-realistic fake images and videos has become fast and effortless. This surge in synthetic content brings serious consequences, including:
- The spread of misinformation and fake news
- Defamation and violations of personal privacy
- Erosion of trust in digital media

As these manipulated visuals become increasingly convincing and widespread, there is a growing need for robust detection systems—especially ones capable of identifying fakes even after compression,
a common practice on social media platforms that further obscures signs of tampering.

## 🌎 Execution Roadmap
<img width="1235" height="559" alt="image" src="https://github.com/user-attachments/assets/d018ea37-1f56-42d1-b6f3-b2c56f3e0287" />

## 📑 Dataset
Used Dataset : [FaceForensics++](https://github.com/ondyari/FaceForensics/tree/master/dataset/DeepFakeDetection)

***⚠️ **Note :** Although the complete dataset is extensive, due to computational and resource constraints, we utilized only a representative subset for training and evaluation.***

Dataset we used for Model Training : [Deep Fake Detection (DFD) Dataset](https://www.kaggle.com/datasets/sanikatiwarekar/deep-fake-detection-dfd-entire-original-dataset)

    It contains total of 3432 videos, out of which 3068 are manipulated and 364 are original.


## ⚙️ Models
### 1. InceptionV3
  - It has proven performance on image classification benchmarks, making it reliable for identifying manipulated content.
  
  - Capable of handling varied input resolutions and distortions, including compression artifacts common in social media videos.
  
  - Easy to fine-tune with transfer learning, reducing training time and improving convergence on limited deepfake datasets.

### 2. ResNet18
  - It prevents vanishing gradients, enabling effective training even at 18 layers deep.
  
  - It offers a great trade-off between speed and accuracy, that makes it ideal for real-time use.
  
  - It performs well on various datasets, especially with pre-trained weights (e.g., ImageNet).


## 🧑‍💻 Model Training
Out of total 16793 images, we have considered the ratio of 0.7 for training set, 0.15 for validation set and remaining 0.15 for testing set. That results in:
 1. 11755 Training Images
 2. 2519 Validation Images
 3. 2519 Testing Images

### Image Transformation:
 - For **IceptionV3 :** Image resolution is converted to 299 x 299
 - For **ResNet18 :** Image resolution is converted to 224 x 224

we normalized the images using the mean and std. deviation.

### Optimizer: ***Adam***
- It combines the advantages of AdaGrad and RMSProp, adapting the learning rate for each parameter.
- It offers fast convergence and works well for noisy or sparse gradients

### Scheduler: ***ReduceLROnPlateau***
- It Monitors validation loss and reduces the learning rate when it stops improving.
- By this, it helps the model escape local minima and fine-tune performance in later training stages.

### Loss Function: ***Binary Cross Entropy Function***
<img width="708" height="128" alt="image" src="https://github.com/user-attachments/assets/07430da2-f53a-4f5c-be4f-c5f9352b8d7f" />


- Where, **$y_i$** is the ground truth for sample i
- **$\hat{y}_i$** hat is the predicted probability of the sample being in the class i.

## 🔮 Extended Work
- Apart from this, we have used a technique of remote PhotoPlethysmoGraphy (rPPG) to detect the DeepFake videos.
- PPG is a non-invasive method used to measure changes in blood volume using light absorption.
- It is typically extracted from skin regions in videos, especially the face, where color changes reflect heartbeats.

### PPG in detecting DeepFakes
- Real videos contain natural PPG signals, visible as subtle color fluctuations in skin caused by blood flow.
- Deepfake videos often lack consistent or realistic PPG signals because generative models don't accurately replicate physiological cues.
- Effective even under compression or low resolution

### Model Training for PPG
We have considered total 768 videos to train our model, instead of the whole dataset, because of the time and resources contraint.\
Out of total 768 videos, we have considered the ratio of 0.7 for training set, 0.15 for validation set and remaining 0.15 for testing set. That results in:
1. 537 Training Images
2. 115 Validation Images
3. 116 Testing Images

### Optimizer: ***AdamW***
- The model uses AdamW optimizer, which is an improved version of Adam.
- It implements weight decay regularization.

### Scheduler: ***CosineAnnealingLR***
- The learning rate follows a cosine curve, decreasing gradually and then increasing slightly before decreasing again
- This cycling helps the model escape local minima and find better optima

### Loss Function: ***Focal Loss***
<img width="1084" height="169" alt="image" src="https://github.com/user-attachments/assets/4527dc3a-e021-45b3-8df9-dd537520c56e" />


## 👥 Contributors

**Darpan Lunagariya :** [GitHub](https://github.com/dp1405) | [LinkedIn](https://www.linkedin.com/in/darpan-lunagariya-264481288)\
**Neel Patel :** [GitHub](https://github.com/Neel7125) | [LinkedIn](www.linkedin.com/in/neelpatel075)









