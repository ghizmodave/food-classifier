# food-classifier

**food-classifier** is a food image recognition web app based on the [Food-101 dataset](http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz).  
 This repository contains the code developed for:
 * Fine tuning a pre-trained [50-layer ResNet](https://arxiv.org/pdf/1512.03385.pdf) model in [Pytorch](https://pytorch.org), using [Amazon Sagemaker](https://aws.amazon.com/sagemaker/), [Paperspace](https://www.paperspace.com/) and local resources (with a training process resumed in multiple sessions).
 * Deploying the model to an AWS endpoint
 * Prototyping the web app using [Flask](https://flask.palletsprojects.com/en/1.1.x/)
 * Deploying the web app with [heroku](https://www.heroku.com)

Have fun trying food-classifier on the website [foodimage-classifier](https://foodimage-classifier.herokuapp.com) !

> **Note**: after 15 minutes of inactivity, the app is suspended and it will restart at the next call; there might be a bit of delay running the app for the first time. In some case there is also some problem loading the page, just refresh it in case.
 
### Data citation
The dataset used for training the CNN is retrieved from [Food-101 - Mining Discriminative Components with Random Forests](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) using as a reference the relative [research paper](https://data.vision.ee.ethz.ch/cvl/mguillau/publications/Bossard2014eccv.pdf).

>   Lukas Bossard, Matthieu Guillaumin, Luc Van Gool - Food-101 – Mining Discriminative Components with Random Forests

The Food-101 data set consists of images from Foodspotting [1]. Any use beyond scientific fair use must be negotiated with the respective picture owners according to the Foodspotting terms of use [2].  
[1] http://www.foodspotting.com/   
[2] http://www.foodspotting.com/terms/

## Downloading the dataset

Food-101 is a large dataset of about 10 GB that can be downloaded and extracted from a .tar file (having compressed size of 4.7 GB) using the code below:
```
%mkdir ../data
!wget -O ../data/food-101.tar.gz http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz
!tar -zxf ../data/food-101.tar.gz -C data
```

## Training, testing and deploying the model

Training and testing are organised using the following Jupyter Notebooks:  
* **Local training notebooks**, calling functions from pytorch_utils.py
	* **food-classifier_mini_experiment.ipynb** : notebook used for prototyping the training, validation and testing processes reducing the number of classes to predict and observing how model performances change increasing the number of classes.
	* **food-classifier_resume_training.ipynb** : notebook used for training the model saving training info in order to resume the training on a later stage
* **AWS training and deployment notebooks and source files**
 	* **food-classifier_sagemaker_uploaddata.ipynb** : notebook used for retrieving, organising and uploading training data on S3 bucket
	* **food-classifier_sagemaker_train.ipynb** : notebook used for training the model using AWS Sagemaker
	* **food-classifier_sagemaker_deploy.ipynb** : notebook used for model deployment to an endpoint
	* **pytorch source files** for AWS contained in the folder pytorch_source

## Website

The web app has been prototyped using Flask and deployed using heroku with the code contained in 'website' and structured in the following way:
```
website
   |-------app.py
   |-------Procfile
   |-------runtime.txt
   |-------requirements.txt
   |-------storage
   |
   |-------templates
   |	      |----------index.html
   |          |----------pred.html
   |
   |-------pytorch_scripts
              |----------__init__.py
              |----------predict.py
              |----------process.py
              |----------prod_model
   
```
