# SONAR-ROCK-VS-MINE-BINARY_CLASSIFICATION-PROJECT-1-
SONAR(ROCK VS MINE) BINARY_CLASSIFICATION

************************************************************************************
Binary Classification Sonar Project 1 for the Navy: 
Mines vs. Rocks


In this project you will discover how to effectively use the Keras library in your machine learning project by working through a binary classification project step-by-step.

After completing this project, you will know:

•	How to load training data and make it available to Keras.
•	How to design and train a neural network for tabular data.
•	How to evaluate the performance of a neural network model in Keras on unseen data.
•	How to perform data preparation to improve skill when using neural networks.
•	How to tune the topology and configuration of neural networks in Keras.



# Description of the Dataset
The dataset we will use in this tutorial is the Sonar dataset.

This is a dataset that describes sonar chirp returns bouncing off different services. The 60 input variables are the strength of the returns at different angles. It is a binary classification problem that requires a model to differentiate rocks from metal cylinders.

You can learn more about this dataset on the UCI Machine Learning repository:
https://archive.ics.uci.edu/ml/datasets/Connectionist+Bench+(Sonar,+Mines+vs.+Rocks) 

We have download the dataset for free and placed it in the project directory with the filename sonar.csv. You can also directly download the dataset:
https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data 

It is a well-understood dataset. All of the variables are continuous and generally in the range of 0 to 1. The output variable is a string “M” for mine and “R” for rock, which will need to be converted to integers 1 and 0.

A benefit of using this dataset is that it is a standard benchmark problem. This means that we have some idea of the expected skill of a good model. Using cross-validation, a neural network should be able to achieve performance around 84% with an upper bound on accuracy for custom models at around 88%.


# What we achieved doing this Project for the Navy:
In this post, you discovered the three API styles used in Keras Deep Learning library in Python.

You learned how you can work through a binary classification problem step-by-step with Keras, specifically:

•	How to load and prepare data for use in Keras.
•	How to create a baseline neural network model.
•	How to evaluate a Keras model using scikit-learn and stratified k-fold cross validation.
•	How data preparation schemes can lift the performance of your models.
•	How experiments adjusting the network topology can lift model performance.



