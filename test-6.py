import requests

url = "http://127.0.0.1:8080/predict"
file_path = "6- Deploying the Speech Recognition System with uWSGI/test/down.wav"

with open(file_path, "rb") as f:
    files = {"file": f}
    response = requests.post(url, files=files)

print(response.status_code)
print(response.text)
