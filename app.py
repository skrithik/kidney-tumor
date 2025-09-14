from flask import Flask, request, jsonify, render_template
import os
from flask_cors import CORS, cross_origin
from cnnClassifier.utils.common import decodeImage
from cnnClassifier.pipeline.prediction import PredictionPipeline


#Ensures proper encoding (UTF-8), avoids locale issues.
os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)
# Creates Flask app.

# Enables CORS → allows frontend apps (like React/Angular) to talk to this API.

class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"#filename = where uploaded image will be saved.
        self.classifier = PredictionPipeline(self.filename)#classifier = instance of PredictionPipeline, which uses the trained model.


@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')#When you open the API root (/), it serves index.html (frontend page).




@app.route("/train", methods=['GET','POST'])
@cross_origin()
def trainRoute():
    os.system("python main.py")
    #os.system("dvc repro")
    return "Training done successfully!"
#Endpoint: /train
#uns main.py → triggers full training pipeline (Stages 1 → 4).



@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    image = request.json['image']
    decodeImage(image, clApp.filename)##Expects JSON with base64-encoded image.Decodes and saves to inputImage.jpg.
    result = clApp.classifier.predict()#Passes file to PredictionPipeline.predict().
    return jsonify(result)#Returns prediction result (Tumor or Normal) as JSON.
#Endpoint: /predict


if __name__ == "__main__":
    clApp = ClientApp()

    app.run(host='0.0.0.0', port=8080) #for AWS

