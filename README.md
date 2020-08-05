# MaskedOrNot
This is a mask detection project which was inspired from the ongoing pandemic
The project can be divided into two parts -

 1. Face Detection
 2. Mask Detection
 
## 1. Face Detection
The model that I used here for the face detection part can be found [here](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/blob/master/models/onnx/version-RFB-640.onnx)
The mechanism behind the face detection can easily be explained from the picture below
![](https://github.com/mehreen-r6234/MaskedOrNot/blob/master/images/face-detection.png)
 
## 2. Mask Detection
The model that I used here for the mask detection part can be found [here](https://github.com/estebanuri/facemaskdetector/blob/master/android/app/src/main/assets/mask_detector.tflite)
The mechanism behind the mask detection can easily be explained from the picture below :
![](https://github.com/mehreen-r6234/MaskedOrNot/blob/master/images/mask-detection.png)

## Tested the environment that works 
 - Memory		31.3 GiB 
 - Processor	Intel® Core™ i5-4590 CPU @ 3.30GHz × 4
 - OS		Ubuntu 18.04.3 LTS 
 - OS type		64-bit 

## Dataset
For evaluation, a testset containing total 212 images were prepared among which 106 images are of masked faces and the other 106 images are of bare faces. These images were carefully collected from [shutterstock](https://www.shutterstock.com/home) and [unsplash](https://unsplash.com/) and renamed(mask_00000.jpg, no-mask_00000.jpg) accordingly.
## Running the application
1. Clone the project to your desired directory
```
$ git clone 
```
2. Create virtual environment 
```
$ sudo apt install python3-dev
$ mkdir ~/venvs
$ python3 -m venv ~/venvs/mask_detection
$ source ~/venvs/mask_detection/bin/activate
$ pip install --upgrade pip
```
3. Install Python Packages
`requirements.txt` contains all the required python packages to run this project 
```
$ pip install -r requirements.txt
```
4. Run the application in a screen

```
$ screen -S mask_detection
$ source ~/venvs/mask_detection/bin/activate
$ export PORT=port && python app.py # if PORT not exported, default is 5000
[Press Ctrl+A+D to detach from screen]
```

## Evaluation

After running the application, run `evaluate.py` to get performance evaluation output
```
$ python evaluate.py --dirpath path/to/dir_of_images --port PORT # default port '5000'
```
The output for the `./dataset` folder :
```
              precision    recall  f1-score   support

     no-mask       0.98      0.98      0.98       106
        mask       0.98      0.98      0.98       106

    accuracy                           0.98       212
   macro avg       0.98      0.98      0.98       212
weighted avg       0.98      0.98      0.98       212
```

## Performance testing using locust
After running `app.py`, for `/detect_faces` and `detect_mask` APIs, performance testing is done using locust

1. Run `locustfile.py` using the command below
```
$ env IM_DIR="./dataset/images_all_final/" locust # for different locust filename, use -f flag 
```
After running this successfully, a web interface will start at `http://0.0.0.0:8089/`
![](https://github.com/mehreen-r6234/MaskedOrNot/blob/master/images/ss_1.png)

2. After filling the boxes accordingly, start swarming
![](https://github.com/mehreen-r6234/MaskedOrNot/blob/master/images/ss_2.png)
3. A detailed statistics will be available
![](https://github.com/mehreen-r6234/MaskedOrNot/blob/master/images/ss_4.png)
4. From `charts`, total rps, response time and number of users can be observed in runtime
![](https://github.com/mehreen-r6234/MaskedOrNot/blob/master/images/ss_6.png)
![](https://github.com/mehreen-r6234/MaskedOrNot/blob/master/images/ss_7.png)

## References
- [https://github.com/estebanuri/facemaskdetector](https://github.com/estebanuri/facemaskdetector)
- [https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB)

