from flask import Flask, render_template, request, send_file, redirect, url_for
from src.exception import CustomException
import os
import sys
from src.logger import logging as lg
from src.pipeline.predict_pipeline import PredictionPipeline
from src.pipeline.train_pipeline import TrainingPipeline

app = Flask(__name__)

# Flag to indicate whether training is completed or not
training_completed = False

@app.route("/")
def home():
    if not training_completed:
        return render_template('train_form.html')
    else:
        return redirect(url_for('predict'))

@app.route("/train", methods=['POST'])
def train_route():
    global training_completed
    try:
        train_pipeline = TrainingPipeline()
        train_pipeline.start_training_pipeline()
        training_completed = True

        return redirect(url_for('predict'))  # Redirect to the predict route after training completion
    except Exception as e:
        raise CustomException(e, sys)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    try:
        if request.method == 'POST':
            # it is an object of prediction pipeline
            prediction_pipeline = PredictionPipeline(request)

            # now we are running this run pipeline method
            prediction_file_detail = prediction_pipeline.run_pipeline()

            lg.info("prediction completed. Downloading prediction file.")
            return send_file(prediction_file_detail.prediction_output_path,
                             download_name=prediction_file_detail.prediction_output_filename,
                             as_attachment=True)
        else:
            # Render the predict_form.html for GET requests
            return render_template('predict_form.html')
    except Exception as e:
        raise CustomException(e, sys)

# Code execution starts here
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)