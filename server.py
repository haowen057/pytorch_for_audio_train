import random
import os
from flask import Flask, request, jsonify
from keyword_spotting_service import Keyword_Spotting_Service


# instantiate flask app
app = Flask(__name__)


# 定义路径,只接受POST请求
@app.route("/predict", methods=["POST"])

def predict():

	# get file from POST request and save it
	audio_file = request.files["file"]
	file_name = str(random.randint(0, 100000))
	audio_file.save(file_name)

	# instantiate keyword spotting service singleton and get prediction
	kss = Keyword_Spotting_Service()
	predicted_keyword = kss.predict(file_name)

	# we don't need the audio file any more - let's delete it!
	os.remove(file_name)

	# send back result as a json file
	result = {"keyword": predicted_keyword}
	return jsonify(result)


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=False)


# 使用绝对路径预测:

# curl -X POST -F "file=@C:\Users\14217\Desktop\Deep-Learning-Audio-Application-From-Design-to-Deployment/6- Deploying the Speech Recognition System with uWSGI\test\down.wav" http://127.0.0.1:5000/predict