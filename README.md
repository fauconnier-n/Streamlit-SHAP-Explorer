# Streamlit Web Application
The goal is to predict the genre of a song, interactively explore the corresponding SHAP values and locally explain a CatBoost Multi Classification model.

**Web application made with [Streamlit](https://streamlit.io/), deployed on an AWS EC2 t2.micro instance with Docker**  
The docker image can be found on [DockerHub](https://hub.docker.com/repository/docker/nfauco/streamlit-shap).  
The dockerfile used to build the image can be found at the root of this repository.  

## Access the Web Application: [~~HERE~~ (not hosted anymore)](http://)  
![](https://github.com/fauconnier-n/Streamlit-SHAP-Explorer/blob/main/images/MyApp.jpg) 
*- the sidebar on the left contains fields to input the features of a song and get the corresponding genre predictions outputed by the model*  
*- the first two plots use the SHAP package to plot the Shapley values and feature importance of the inputed song*  
*- the third one allows to select two features and a genre to show the corresponding Dependence Plot of the Shapley values of every predictions made on the evaluation dataset*  

***

### The model
In the **[notebook](https://github.com/fauconnier-n/Streamlit-SHAP-Explorer/blob/main/notebook.ipynb)**, Cross Validation and GridSearch are used for hyperparameters optimization.  
However, the app isn't running the model offering the best accuracy, but a model trained with way less iterations. It makes the predictions are faster (less trees), and the model is less prone to overfitting.

Best Hyperparameters found (~58% Accuracy):  
*{'depth': 6,  
 'iterations': 1000,  
 'learning_rate': 0.1}*  
 
Model used in the app (~54% Accuracy):  
*{'depth': 6,  
 'iterations': 50,  
 'learning_rate': 0.125}*  

***

### More plots from the notebook for *song genre = EDM*
![](https://github.com/fauconnier-n/Streamlit-SHAP-Explorer/blob/main/images/beeswarm%20edm.png)  
*Beeswarm plot for genre = EDM : shows feature importance in every prediction made on the evaluation set*  

![](https://github.com/fauconnier-n/Streamlit-SHAP-Explorer/blob/main/images/dependence_plots.png)  
*More dependence plots, all accessible one by one on the app*  
