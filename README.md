# openvino-demo
Simple to follow OpenVino demo for new comers

## Clone Repo
```
git clone https://github.com/yockgen/openvino-demo.git
```
```
cd openvino-demo
```
## Setup Virtual Environment
OpenVino required many dependecies software and may conflict with user's existing Python libs and usually costed time to debug, created virtual is the safer way to avoid such conflict
```
python -m venv openvino
```
```
 .\openvino\Scripts\activate
```
If succeed, user shall see prompt with venv prefix, example as below:
```
 (openvino) PS C:\openvino-demo>
```
## Install pre-requisties
```
pip install -r requirements.txt
```
## Run Test
```
python face.py
```
If succeed, the script will access first webcam of your device, and start detect faces, emotions, gender and age.
