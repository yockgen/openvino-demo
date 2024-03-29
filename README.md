# Intel OpenVino Demo
A straightforward OpenVino demonstration suitable for newcomers. This demo utilizes the device's webcam to detect faces, emotions, age, and gender.

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
## CPU or GPU
The script by default is expected to run on Intel Integrated GPU, if user would like to change to CPU, please modify accordingly "device_name" in "face.py" (3 of them): 
```
 self.compiled_model = ie.compile_model(model=model, device_name="GPU")
```

## Run Test
```
python face.py
```
If succeed, the script will access first webcam of your device, and start detect faces, emotions, gender and age.
![Alt text](demo.jpg)
