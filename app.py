#!/usr/bin/env python
# coding: utf-8

# In[13]:


import numpy as np
from flask import Flask,request,render_template
import pickle
loaded_model = pickle.load(open('model_pickle.pkl','rb')) 
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')
    
@app.route('/p',methods=['POST'])
def predict():
    # getting the values of independent variables from server
    if request.method=="POST":
        output = request.form.to_dict()
        Month = output['MonthValue']
        DayofMonth =output['DayofMonthValue']
        DayofWeek  = output['DayofWeekValue']
        Destination = output['DestiValue']
        DepDelay = output['DepDelayValue']
        SchDepTime = output['SchDepTimeValue']
        SchArrTime  = output['SchArrTimeValue']
        Duration   = output['DurationValue']
        TotalScheduled  = output['TotalScheduledValue']
        AvgSpeed  = output['AvgSpeedValue']
        Temperature  = output['TemperatureValue']
        Pressure  = output['PressureValue']
        Dew  = output['DewValue']
        Humidity  = output['HumidityValue']
        WindSpeed  = output['WindSpeedValue']
        Wind  =output['WindValue']
        WindGust  = output['WindGustValue']
        Condition=output['ConditionValue']
        
        #converting these values into list
        ValueArray = list([ Month, DayofMonth, DayofWeek, Destination, DepDelay, Duration, SchDepTime, SchArrTime, Temperature, Dew, Humidity, Wind, WindSpeed, WindGust, Pressure, Condition, TotalScheduled, AvgSpeed ])
    
        # converting the time from(HH:MM) to minutes (0 to 14400)        
        ValueArray[6] = (int(ValueArray[6].split(':')[0]) * 60) + (int(ValueArray[6].split(':')[1]) * 1)
        ValueArray[7] = (int(ValueArray[7].split(':')[0]) * 60) + (int(ValueArray[7].split(':')[1]) * 1)
        
        # converting to array
        ValueArray=list(map(float,ValueArray))
        ValueArray = np.array(ValueArray).reshape(1,-1).flatten()
        
        # min max scaling
        ValueArray[1] = ((ValueArray[1] * 1) - 1) / (31 - 1)
        ValueArray[2] = ((ValueArray[2] * 1) - 1) / (7 - 1)
    
        # transformations
        ValueArray[4] = np.cbrt(ValueArray[4] * 1)
        ValueArray[5] = np.log(ValueArray[5] * 1)
        ValueArray[16] = np.power((ValueArray[16] * 1), 3)
    
        # Standard scaling
        ScaleArray = [[ 0, 0, 0, 0, -4.48541610e-01,  5.26745748e+00,8.30488227e+02, 9.10254381e+02,  4.15495108e+01,  3.06773465e+01, 5.81878651e+01, 0, 1.20508547e+01, 0, 3.00918310e+01, 0, 2.38756486e+05,  4.99314984e+00],[ 1, 1, 1, 1, 1.98131375e+00,  5.65632200e-01,  2.99871664e+02, 3.45860197e+02,  7.76191155e+00,  1.19208650e+01, 2.33795502e+01, 1, 5.93881074e+00, 1, 2.91169450e-01, 1, 1.18471648e+05,  1.42228659e+00]]  # mean and std
        
        for i in ([3,4,5,6,7,8,9,10,11,12,14,15,16,17]):
            ValueArray[i] = ((ValueArray[i] * 1) - ScaleArray[0][i]) / (ScaleArray[1][i])
            
        ValueArray=ValueArray.reshape(1,18)
            
        #predicting using the loaded model
        result = loaded_model.predict(ValueArray)
        if int(result)==1:
            prediction='Taxi out delay is more than 20 mins'
        elif int(result)==0:
            prediction='Taxi out delay is less than 20 mins'
        return render_template('index.html',pred=prediction)

        
if __name__=='__main__':
    app.run(debug=True,use_reloader=False)


# In[ ]:




