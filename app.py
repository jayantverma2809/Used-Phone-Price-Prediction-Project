from flask import Flask, render_template, request
import pickle as pk
import numpy as np
import requests

app = Flask(__name__,static_folder='statics')
model = pk.load(open('Price_Predictor.pkl','rb')) 
scaler = pk.load(open('Scaler.pkl','rb'))

@app.route('/',methods=['GET']) 
def home(): 
    return render_template('index.html')

@app.route('/predict', methods=['POST']) 
def predict(): 
    if request.method == 'POST': 
        screeen_size = float(request.form['screen_size'])
        screeen_size = np.log(screeen_size)
        FourG = request.form['4g']
        if FourG == "Yes":
            FourG=1
        else:
            FourG=0
        FiveG = request.form['5g']
        if FiveG == "Yes":
            FiveG = 1
        else:
            FiveG = 0
        rear_camera_mp = float(request.form['rear_camera_mp'])
        rear_camera_mp = rear_camera_mp**(1/2) 
        front_camera_mp = float(request.form['front_camera_mp'])
        front_camera_mp = np.log(front_camera_mp)
        internal_memory = int(request.form['internal_memory'])
        ram = float(request.form['ram'])
        battery = int(request.form['battery'])
        battery = np.log(battery)
        weight = float(request.form['weight'])
        weight = 1/weight
        release_year = request.form['release_year']
        year2014,year2015,year2016,year2017,year2018,year2019,year2020 = 0,0,0,0,0,0,0
        if release_year=='2014':
            year2014 = 1
        elif release_year=='2015':
            year2015 = 1
        elif release_year=='2016':
            year2016 = 1
        elif release_year=='2017':
            year2017 = 1
        elif release_year=='2018':
            year2018 = 1
        elif release_year=='2019':
            year2019 = 1
        elif release_year=='2020':
            year2020 = 1
        days_used = int(request.form['days_used'])
        new_price = float(request.form['normalized_new_price'])
        normalized_new_price = np.log(new_price)
        brand_dict = {'Acer': 4.294424174,'Alcatel': 4.026422546,'Apple': 5.011901159,'Asus': 4.4673986215,'BlackBerry': 4.2931666795000005,
                      'Celkon': 3.116621591,'Coolpad': 4.243339115,'Gionee': 4.349177705,'Google': 4.870146421,
                      'HTC': 4.432936787999999,'Honor': 4.6833263,'Huawei': 4.69015451,'Karbonn': 3.654546775,
                      'LG': 4.322143925,'Lava': 3.9323964145000003,'Lenovo': 4.423648309,'Meizu': 4.533451438,
                      'Micromax': 3.867443962,'Microsoft': 4.195245147,'Motorola': 4.38256042,'Nokia': 4.052654135,
                      'OnePlus': 4.679163866,'Oppo': 4.69701984,'Others': 4.2107191964999995,'Panasonic': 4.282206299,
                      'Realme': 4.668802046,'Samsung': 4.51008998,'Sony': 4.527100531,'Spice': 3.6704561995000002,
                      'Vivo': 4.761831996,'XOLO': 3.947337803,'Xiaomi': 4.630935394,'ZTE': 4.360214212500001}
        Device_Brand = request.form['Device_Brand']
        Device_Brand = brand_dict[Device_Brand]
        
        test_set = np.array([[screeen_size,FourG,FiveG,rear_camera_mp,front_camera_mp,internal_memory,
                                    ram,battery,weight,days_used,normalized_new_price,Device_Brand,
                                    year2014,year2015,year2016,year2017,year2018,year2019,year2020]])
        test_set = scaler.transform(test_set)
        prediction = model.predict(test_set)
        if new_price<=10000:
            prediction = np.exp(prediction)*10
        elif new_price>10000 and new_price<=20000:
            prediction = np.exp(prediction)*20
        elif new_price>20000 and new_price<=30000:
            prediction = np.exp(prediction)*25
        elif new_price>30000 and new_price<=50000:
            prediction = np.exp(prediction)*35
        elif new_price>50000 and new_price<=70000:
            prediction = np.exp(prediction)*40
        elif new_price>70000 and new_price<=1000000:
            prediction = np.exp(prediction)*50

        return render_template('index.html', prediction_text="Current price of phone is Rs.{}".format(prediction))
    else:
        return render_template('index.html')
 
if __name__ == '__main__': 
    app.run(debug=True)
