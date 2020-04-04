food-classifier
===============

This repo contains the work done for developing a food image recognition web app deployed using [heroku](https://www.heroku.com) on the website [foodimage-classifier](https://foodimage-classifier.herokuapp.com). 

(Note: after 15 minutes of inactivity, Heroku will suspend the app, restarting it at the next call; there might be a bit of delay starting the app for the first time. In some case there is also some problem loading the page, just refresh it in case).

[Pytorch](https://pytorch.org) has been used for implementing the convolutional neural network (CNN) model, trained using [Amazon Sagemaker](https://aws.amazon.com/sagemaker/) and local resources (with the training process resumed in multiple sessions.

 

Data citation
-------------

The dataset used for training the CNN is retrieved from [Food-101 - Mining Discriminative Components with Random Forests](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) using as a reference the relative [research paper](https://data.vision.ee.ethz.ch/cvl/mguillau/publications/Bossard2014eccv.pdf) from:

>   Lukas Bossard, Matthieu Guillaumin, Luc Van Gool - Food-101 – Mining Discriminative Components with Random Forests

The Food-101 data set consists of images from Foodspotting [1]. Any use beyond scientific fair use must be negotiated with the respective picture owners according to the Foodspotting terms of use [2].

[1] http://www.foodspotting.com/   
[2] http://www.foodspotting.com/terms/
