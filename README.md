# food-classifier

**food-classifier** is a food image recognition web app based on the [Food-101 dataset](http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz).  
 This repository contains the code developed for:
 * Fine tuning a pre-trained [50-layer ResNet](https://arxiv.org/pdf/1512.03385.pdf) model in [Pytorch](https://pytorch.org), using [Amazon Sagemaker](https://aws.amazon.com/sagemaker/), [Paperspace](https://www.paperspace.com/) and local resources (with a training process resumed in multiple sessions).
 * Deploying the model to a AWS endpoint
 * Prototyping the web app using [Flask](https://flask.palletsprojects.com/en/1.1.x/)
 * Deploying the web app with [heroku](https://www.heroku.com)

Have fun trying food-classifier on the website [foodimage-classifier](https://foodimage-classifier.herokuapp.com)!

> **Note**: after 15 minutes of inactivity, the app is suspended and it will restart at the next call; there might be a bit of delay running the app for the first time. In some case there is also some problem loading the page, just refresh it in case.
 
#### Data citation
-------------

The dataset used for training the CNN is retrieved from [Food-101 - Mining Discriminative Components with Random Forests](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) using as a reference the relative [research paper](https://data.vision.ee.ethz.ch/cvl/mguillau/publications/Bossard2014eccv.pdf).

>   Lukas Bossard, Matthieu Guillaumin, Luc Van Gool - Food-101 – Mining Discriminative Components with Random Forests

The Food-101 data set consists of images from Foodspotting [1]. Any use beyond scientific fair use must be negotiated with the respective picture owners according to the Foodspotting terms of use [2].  
[1] http://www.foodspotting.com/   
[2] http://www.foodspotting.com/terms/

