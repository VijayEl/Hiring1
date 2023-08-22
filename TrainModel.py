from flask import Flask, request, url_for, redirect, render_template,send_from_directory,make_response
from Functions import classificationModel
import pandas as pd
from fileinput import filename
import pickle

#trainData = pd.read_excel('C:/HiringEngine/Data/RenegeData.xlsx')
#testData = pd.read_excel('C:/Users/lgorle/Desktop/flaskapi/HiringEngine_Flask/Data/TestData.xlsx')
#Finalmodel=pd.read_pickle('C:/HiringEngine/savedmodel/Randomforest.pkl')

cmobj=classificationModel.classificationModel()
#rf_clf=cmobj.RF(trainData)
#df=cmobj.preProcesstestData(test_data[:2000],model)

app = Flask(__name__)

# Root endpoint
@app.get('/')
def upload():
	return render_template('Train.html')
    

@app.post('/view')
def view():
 
    # Read the File using Flask request
    file = request.files['file']
    # save file in local directory
    file.save(file.filename)
 
    # Parse the data as a Pandas DataFrame type
    traindata = pd.read_excel(file)
    traindatacopy=traindata.copy()
    rf_clf,X_train_rf_clf,X_test_rf_clf,y_train_rf_clf,y_test_rf_clf=cmobj.RF(traindata)
    train_result,test_result=cmobj.results(rf_clf,X_train_rf_clf,X_test_rf_clf,y_train_rf_clf,y_test_rf_clf)
    accuracy_scores,crossvalidation_scores=cmobj.Qualitycheck(traindatacopy)
    print(train_result)
    print(test_result)
    print(accuracy_scores)
    print(crossvalidation_scores)
    resultdata=pd.DataFrame()
    #resultdata['Accuracy on Train Data']=list(train_result)
    #resultdata['Accuracy on Test Data']=list(test_result)
    #resultdata['Accuracies on 10 test sets']=list(accuracy_scores)
    #resultdata['cross_validation']=list(crossvalidation_scores)
    #resultdata.to_excel('resultdata.xlsx',index=False)
    #save the iris classification model as a pickle file
    model_pkl_file = "trained_model.pkl"  

    with open(model_pkl_file, 'wb') as file:  
        pickle.dump(rf_clf, file)
    return "Model downloaded"
    # write dataframe to excel
    #resultdata.to_excel('resultdata.xlsx',index=False)
  
    #return send_from_directory("C:/HiringEngine","resultdata.xlsx")
        
if __name__ =="__main__":
    app.run()
