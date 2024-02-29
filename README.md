# CustomVision - ONNX Object Detection

Created by exporting the customvision model as zip file and sample code.

-- When using a Raspberry Pi with a Raspberry Pi Camera, due to OpenCV (cv2), it requires legacy camera to be enabled in the raspi-config.

## Start with a virtual environment

```
cd
python -m venv envCustomVisionOnnx
. envCustomVisionOnnx/bin/activate
```

## Clone the repo then enter the folder

```
git clone https://github.com/NoliAlonso/CustomVisionONNX
cd CustomVisionONNX
```

## Installation

```
pip install -r requirements.txt
```

## Run using:

```
python onnxruntime_predict.py
```

Expected performance is 1 to 2 FPS. Noticeably faster with the float16 model.
Toggle detection by touching the window, allows much better camera movement as the FPS goes up to 30.
