from flask import Flask,request,render_template,jsonify
from src.pipelines.prediction_pipeline import CustomData,PredictPipeline
from src.exception import CustomException
import os
import sys

application=Flask(__name__)

app=application



@app.route('/')
def home_page():
    try:
        return render_template('index.html')
    except Exception as e:
        raise CustomException(e,sys)

@app.route('/predict',methods=['GET','POST'])


def predict_datapoint():
    if request.method=='GET':
        return render_template('form.html')
    
    else:

        data=CustomData(
            occ_2=float(request.form.get('occ_2')),
            occ_3 = float(request.form.get('occ_3')),
            occ_4 = float(request.form.get('occ_4')),
            occ_5 = float(request.form.get('occ_5')),
            
            occ_6= float(request.form.get('occ_6')),
            occ_husb_2= float(request.form.get('occ_husb_2')),
            occ_husb_3= float(request.form.get('occ_husb_3')),
            occ_husb_4= float(request.form.get('occ_husb_4')),
            occ_husb_5 = float(request.form.get('occ_husb_5')),

            occ_husb_6 =float (request.form.get('occ_husb_6')),
            rate_marriage= float(request.form.get('rate_marriage')),
            yrs_married =float( request.form.get('yrs_married')),
            children = float(request.form.get('children')),
            religious= float(request.form.get('religious')),
            educ = float(request.form.get('educ')))


        final_new_data=data.get_data_as_dataframe()
        print(final_new_data)
        predict_pipeline=PredictPipeline()
        pred=predict_pipeline.predict(final_new_data)

        results=round(pred[0],2)

        return render_template('result.html',final_result=results)






if __name__=="__main__":
    app.run(host='0.0.0.0',debug=True)