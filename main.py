import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from flask import Flask, render_template, request 
import warnings 
warnings.filterwarnings('ignore') 

app = Flask(__name__) 

@app.route('/', methods=["GET", "POST"]) 
def hello_world(): 
    if request.method == "POST": 
        myDict = request.form 
        
        N = float(myDict['nitrogen']) 
        P = float(myDict['phosphorous']) 
        K = float(myDict['potassium']) 
        temperature = float(myDict['temperature']) 
        ph = float(myDict['ph']) 
        humidity = float(myDict['humidity']) 
        rainfall = float(myDict['rainfall']) 
        
        data = pd.read_csv("Crop_recommendation.csv") 
        
        y = data['label'] 
        x = data.drop(['label'], axis = 1) 
        
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0) 
        
        model = LogisticRegression(solver='liblinear')
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        inputfeatures = [N,P,K,temperature,ph,humidity,rainfall] 
        prediction = model.predict([inputfeatures]) 
        print("The Suggested Crop for Given Climatic Condition is :", prediction) 
        
    
        return render_template('show.html', inf=prediction) 
    return render_template('home.html') 

@app.route('/fert') 
def fert(): 
    return render_template('fert.html') 
 
@app.route('/home') 
def home(): 
    return render_template('home.html') 
 
@app.route('/predict') 
def predict(): 
    return render_template('predict.html') 
 
@app.route('/about') 
def about(): 
    return render_template('about.html') 

if __name__ == "__main__": 
    app.run(debug=True)