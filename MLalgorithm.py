################################################import libraries###################################################
import csv
import warnings
import cv2
import numpy as np
import dlib
from imutils import face_utils
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import math
import numpy as np
import argparse
import cv2
from similaritymeasures import similaritymeasures
from sklearn.multiclass import OneVsRestClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from sklearn.svm import SVC
import matplotlib.pyplot as plt
################################################define variables###################################################
counter=0
ArrayX = np.empty((13,), dtype=object)
ArrayY = np.empty((13,), dtype=object)
XX=[]
YY=[]
Coefficients=[]
NumberHappyFace=0
NumberTotallFace=0
######################################################CNN define var###################################################
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# ap = argparse.ArgumentParser()
# ap.add_argument("--mode",help="train/display")
# a = ap.parse_args()
# mode = 'display'
######################################################Function for CNN###################################################
def plot_model_history(model_history):
    """
    Plot Accuracy and Loss curves given the model_history
    """
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    # summarize history for accuracy
    axs[0].plot(range(1,len(model_history.history['acc'])+1),model_history.history['acc'])
    axs[0].plot(range(1,len(model_history.history['val_acc'])+1),model_history.history['val_acc'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_history.history['acc'])+1),len(model_history.history['acc'])/10)
    axs[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')
    fig.savefig('plot.png')
    plt.show()
######################################################Model for CNN###################################################
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))
######################################################Load model###################################################
model.load_weights('model.h5')
cv2.ocl.setUseOpenCL(False)
################################################define Haar and dlib################################################
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


################################################identify video###################################################
DIR = 'Dataset/Train'
#CATEGORIES = ["Azin"]
CATEGORIES = ["Azin","Hadi","Mahsa","Mohammad","Nadia","Neda","Parisa", "Rojan","Shaghayegh","Sister1","Sister2","Soheil","Zahra"]
for CATEGORY in CATEGORIES:
    path = os.path.join(DIR, CATEGORY)  # paths to the legobricks
    class_num = CATEGORIES.index(CATEGORY)
    for Video in os.listdir(path):
        counter=0
        NumberHappyFace=0
        NumberTotallFace=0
        cap = cv2.VideoCapture(os.path.join(path, Video))
        #cap = cv2.VideoCapture('5.avi')
        ##########################################ShiTomasi corner detection###############################################
        feature_params = dict(maxCorners=100,
                              qualityLevel=0.3,
                              minDistance=7,
                              blockSize=7)
        ##########################################lucas kanade optical flow ###############################################
        # Parameters for lucas kanade optical flow
        lk_params = dict(winSize=(15, 15),
                         maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        ##########################################frame preprocessing###############################################
        # Create some random colors
        color = np.random.randint(0, 255, (100, 3))
        # Take first frame and find corners in it
        ret, old_frame = cap.read()
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        height1, width1 = old_frame.shape[:2]
        ##########################################find face in frame## ###############################################
        faces = face_cascade.detectMultiScale(old_gray, 1.1, 1, minSize=(40, 40))
        #faces = face_cascade.detectMultiScale(old_gray, 1.1, 1, minSize=(int(height1/3), int(width1/2)))
        ##########################################frame preprocessing###############################################
        for (x, y, w, h) in faces:
            print("ok")
        #old_crop_img = old_frame[y:y+h, x:x+w]
        old_crop_img = old_frame[y-20:y+h+20, x-20:x+w+20]
        #cv2.imshow('frame', old_crop_img)
        #cv2.waitKey(0)
        height, width = old_frame.shape[:2]
        old_crop_img1=old_crop_img.copy()
        #cv2.imshow('ff',old_crop_img1)
        #cv2.imshow('frame', old_crop_img1)
        #cv2.waitKey(0)
        old_crop_img = cv2.cvtColor(old_crop_img, cv2.COLOR_BGR2GRAY)
        ##########################################fine landmarks by dlib###############################################
        faces = detector(old_crop_img)
        for face in faces:
            print("yes")
            landmarks = predictor(old_crop_img, face)
            shape = face_utils.shape_to_np(landmarks)
            shape = np.array(shape).reshape(-1, 2)
            cheeksleft = (shape[4][0] + int((shape[48][0] - shape[4][0]) / 2),
                                           shape[29][1] + int((shape[33][1] - shape[29][1]) / 2))
            cheeksright = (shape[54][0] - int((shape[54][0] - shape[12][0]) / 2),
                                            shape[29][1] + int((shape[33][1] - shape[29][1]) / 2))
            ntotal = 13
        #     ntotal=10
            k = 0
            array = np.ndarray((ntotal, 1, 2))
            array[k]=cheeksleft
            k+=1
            array[k] = cheeksright
            k += 1
            array[k] = shape[31]
            k += 1
            array[k] = shape[35]
            k += 1
            array[k] = shape[36]
            k += 1
            array[k] = shape[45]
            k += 1
            array[k] = shape[48]
            k += 1
            array[k] = shape[49]
            k += 1
            array[k] = shape[51]
            k += 1
            array[k] = shape[53]
            k += 1
            array[k] = shape[54]
            k += 1
            array[k] = shape[56]
            k += 1
            array[k] = shape[58]
            k += 1
            pp=0
            pp = array.copy()
            pp = np.float32(pp)
            mask = np.zeros_like(old_crop_img1)
            print(pp)
        ##########################################show landmarks on first  frame###########################################
        for i in range(len(pp)):
            XX.append(pp[i][0][0])
            YY.append(pp[i][0][1])
            cv2.circle(old_crop_img, (int(pp[i][0][0]), int(pp[i][0][1])), 4, (255, 0, 0), -1)
        #cv2.imshow('f',old_crop_img)
        #cv2.waitKey(0)
        ###############################################save points in array#################################################
        for i, v in enumerate(ArrayX):
            ArrayX[i] = [pp[i][0][0]]

        for i, v in enumerate(ArrayY):
            ArrayY[i] = [pp[i][0][1]]
        ######################################################start while######################################################
        while (1):
            counter = counter + 1
            if counter <= 200:
                ret, frame = cap.read()
                if ret==1:
                    ##########################################frame preprocessing###############################################
                    crop_img1 = frame[y-20:y+h+20, x-20:x+w+20]
                    height, width = frame.shape[:2]
                    crop_img = cv2.cvtColor(crop_img1, cv2.COLOR_BGR2GRAY)
                    #crop_img = cv2.resize(crop_img, (int(width/2), int(height/2)))
                    #print(crop_img.shape)
                    cv2.imshow('frame', crop_img)
                    cv2.waitKey(2) & 0xff
                    ######################################find and print emotion###############################################
                    mcropped_img = np.expand_dims(np.expand_dims(cv2.resize(crop_img, (48, 48)), -1), 0)
                    mprediction = model.predict(mcropped_img)
                    mmaxindex = int(np.argmax(mprediction))
                    print(emotion_dict[mmaxindex])
                    if emotion_dict[mmaxindex] == "Happy":
                        NumberHappyFace = NumberHappyFace + 1
                    NumberTotallFace = NumberTotallFace + 1
                    ##########################################Calculate optical flow############################################
                    p1, st, err = cv2.calcOpticalFlowPyrLK(old_crop_img, crop_img, pp, None, **lk_params)
                    #############################################Select good points#############################################
                    good_new = p1[st == 1]
                    good_old = pp[st == 1]
                    #############################################draw the tracks################################################
                    for i, (new, old) in enumerate(zip(good_new, good_old)):
                                a, b = new.ravel()
                                c, d = old.ravel()
                                mask = cv2.line(mask, (a, b), (c, d),(120, 20, 222), 2)
                                crop_img1 = cv2.circle(crop_img1, (a, b), 5, (76, 76, 45), -1)
                    img = cv2.add(crop_img1,mask)
                    cv2.imshow('frame', img)
                    cv2.waitKey(100) & 0xff
                    #cv2.waitKey(0)
                    ###############################dNow update the previous frame and previous points############################
                    old_crop_img = crop_img.copy()
                    pp = good_new.reshape(-1, 1, 2)
                    ##########################################save points in array##############################################
                    if len(pp)<13:
                        continue
                    try:
                        for i, v in enumerate(ArrayX): v.append(pp[i][0][0])
                        for i, v in enumerate(ArrayY): v.append(pp[i][0][1])
                    except:
                        print("error")
            if counter > 200:
                break

        cv2.destroyAllWindows()
        cap.release()

        ArrayX=np.asarray(ArrayX)
        print(ArrayX)
        print(len(ArrayX))
        print(type(ArrayX))
        ArrayY=np.asarray(ArrayY)
        print(ArrayY)
        print(len(ArrayY))
        print(type(ArrayY))

        print(NumberHappyFace)
        print(NumberTotallFace)
        #if int(0.2*NumberTotallFace)<=NumberHappyFace:
        if True:
            ###############################################Curve fitting###########################################################
            Coefficients=[]
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', np.RankWarning)
                for i, v in enumerate(ArrayX):
                    p30 = np.poly1d(np.polyfit(ArrayX[i], ArrayY[i], 4))
                    Coefficients.append(p30)
                    print(p30.coef)


            print(Coefficients)
            '''
            def func(x, a, b, c,d,e):
                return a*(math.pow(x, 3))
            popt, pcov = curve_fit(func, arrayX, arrayY)
            print("Sine funcion coefficients:")
            print(popt)
            print("Covariance of coefficients:")
            print(pcov)
            plt.scatter(arrayX, arrayY)
            plt.plot(arrayX, func(arrayX, *popt), 'r-', label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
            plt.xlabel('x')
            plt.ylabel('y')
            plt.legend()
            plt.show()
            '''


        else :
            print("No happy Face")
        ################################################open excel################################################

        with open('Train.csv', 'a') as td:
            for i in range(13):
                co=list((Coefficients[i].coef))
                co.append(class_num)
                td.write(str(co))
                td.write('\n')




