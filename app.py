import numpy as np
from flask import Flask , request , jsonify,render_template
import pickle
from sklearn.ensemble import RandomForestRegressor
import joblib

app=Flask(__name__)
model=joblib.load('finalized_model.sav')

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['Post'])
def predict():

    int_features = [float(x) for x in request.form.values()]
    initial_features =np.array(int_features)
    run_rate=initial_features[3]/initial_features[2]
    
    wickets_left=10-initial_features[4]
    
    initial_features=initial_features[:4]
    initial_features=np.append(initial_features,run_rate)
    initial_features=np.append(initial_features,wickets_left)
    

    
    #AF=(4*OB)+(WR)**3
    Accleration_factor= (4*initial_features[2]) + (wickets_left)**3
    initial_features=np.append(initial_features,Accleration_factor)

    if wickets_left >=6:
         Batting_depth=8
    elif wickets_left >2 and wickets_left <6:
         Batting_depth=6
    else:
         Batting_depth=2
    initial_features=np.append(initial_features,Batting_depth)

    

    Final_features=[initial_features]

    prediction = model.predict(Final_features)

    output = int(prediction[0])
    

    return render_template('predict.html', prediction_text='Predicted Score after 50 Overs : \n {}'.format(output))

if __name__=="__main__":
     app.run(debug=True)