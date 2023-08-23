from flask import Flask, request, url_for, redirect, render_template,send_from_directory,make_response
from Functions import classificationModel
import pandas as pd
from fileinput import filename

#trainData = pd.read_excel('C:/RenegeAnalytics/RenegeAnalytics/Data/RenegeData.xlsx')
trainData = pd.read_excel('Data/RenegeData.xlsx')

traindata=trainData.copy()
#testData = pd.read_excel('C:/Users/lgorle/Desktop/flaskapi/HiringEngine_Flask/Data/TestData.xlsx')
#Finalmodel=pd.read_pickle('C:/HiringEngine/savedmodel/Randomforest.pkl')

cmobj=classificationModel.classificationModel()
rf_clf,X_train_rf_clf,X_test_rf_clf,y_train_rf_clf,y_test_rf_clf=cmobj.RF(trainData)
#df=cmobj.preProcesstestData(test_data[:2000],model)

app = Flask(__name__)

# Root endpoint
@app.get('/')
def upload():
	return render_template('index.html')
    

@app.post('/view')
def view():
 
    # Read the File using Flask request
    file = request.files['file']
    # save file in local directory
    file.save(file.filename)
 
    # Parse the data as a Pandas DataFrame type
    testdata = pd.read_excel(file)
    resultdata=cmobj.preProcesstestData(testdata,rf_clf)
    #resultdata.to_excel('resultdata.xlsx',index=False)
    
    resp = make_response(resultdata.to_csv(index=False))
    resp.headers["Content-Disposition"] = "attachment; filename=Renege_Result.csv"
    resp.headers["Content-Type"] = "text/csv"
    return resp

@app.post('/template')
def template():
    resp1 = make_response(traindata.to_csv(index=False))
    resp1.headers["Content-Disposition"] = "attachment; filename=RawData_Template.csv"
    resp1.headers["Content-Type"] = "text/csv"
    return resp1


        
if __name__ =="__main__":
    # Use Gunicorn as the server
    #from gunicorn.app.wsgiapp import WSGIApplication
    #app_instance = WSGIApplication("%s:app" % __name__)
    #app_instance.run()
    app.run(0.0.0.0', port=80)
