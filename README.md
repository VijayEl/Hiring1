# RenegeAnalytics
Built an application user interface(API) using flask in python to download predicted results of employee's hiring status.
## Folders and Content:
1. Data Folder has required datasets for training and testing putpose
2. Functions Folder has 3 files :
      * preprocess.py - This file has 3 functions such as preProcessing function to preprocess raw data for training, preprocesstestData function to preprocess test data and plot_feature_importance(optional) to plot important features if needed. 
      * EncodingandSplit.py - This file has 2 functions such as labelEncoding to perform labelEncoding for few columns and traintestsplit to split data into train and test datasets.
      * classificationModel.py -This file has 4 functions such as "RF" function which uses random forest classifier to train the model, "preprocesstestData" function to preprocess,label encode and predict results on test data set, "results" function (optional/This has been used while training the model (TrainModel.py)) to display performance metrics such as recall, precision , fi score and accuracy and "Qualitycheck" function to perform crossvalidation and to test model performance on different data sets with in same data.
3. static folder has the image that is used in front end development
4. templates folder has html codes required to create front end application
5. savedmodel folder has 2 trained models in pickle format
6. HiringEngine.py code is to create flask applucation (main code)
7. TrainModel.py does only training (A trained model will be sownloaded)
8. Testmodel.py does only predictions(makes use of trainedmodel which is the result of TrainModel.py
"# Hiring1" 
"# Hire" 
"# Hire" 
