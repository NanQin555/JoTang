#train.py

# 导入packages
import numpy as np
import cv2
import os
import csv

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from keras.optimizers import Adam
from keras.layers import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image

# 将原始数据划分为训练、验证、测试集（虽然只用到了训练/测试集）
database_path = r'C:\Apps\Anconda_new\dataset\\fer2013'
datasets_path = r'C:\Apps\Anconda_new\dataset\\fer2013'
csv_file = os.path.join(database_path, 'fer2013.csv')
train_csv = os.path.join(datasets_path, 'train.csv')
val_csv = os.path.join(datasets_path, 'val.csv')
test_csv = os.path.join(datasets_path, 'test.csv')

# 预处理
with open(csv_file) as f:
    csvr = csv.reader(f)
    header = next(csvr)
    rows = [row for row in csvr]

    trn = [row[:-1] for row in rows if row[-1] == 'Training']
    csv.writer(open(train_csv, 'w+'), lineterminator='\n').writerows([header[:-1]] + trn)
    print(len(trn))

    val = [row[:-1] for row in rows if row[-1] == 'PublicTest']
    csv.writer(open(val_csv, 'w+'), lineterminator='\n').writerows([header[:-1]] + val)
    print(len(val))

    tst = [row[:-1] for row in rows if row[-1] == 'PrivateTest']
    csv.writer(open(test_csv, 'w+'), lineterminator='\n').writerows([header[:-1]] + tst)
    print(len(tst))

    datasets_path = r'C:\Apps\Anconda_new\dataset\\fer2013'
    train_csv = os.path.join(datasets_path, 'train.csv')
    val_csv = os.path.join(datasets_path, 'val.csv')
    test_csv = os.path.join(datasets_path, 'test.csv')

    train_set = os.path.join(datasets_path, 'train')
    val_set = os.path.join(datasets_path, 'val')
    test_set = os.path.join(datasets_path, 'test')

for save_path, csv_file in [(train_set, train_csv), (val_set, val_csv), (test_set, test_csv)]:
    if not os.path.exists(save_path):
            os.makedirs(save_path)

    num = 1
    with open(csv_file) as f:
        csvr = csv.reader(f)
        header = next(csvr)
        for i, (label, pixel) in enumerate(csvr):
            pixel = np.asarray([float(p) for p in pixel.split()]).reshape(48, 48)
            subfolder = os.path.join(save_path, label)
            if not os.path.exists(subfolder):
                os.makedirs(subfolder)
            im = Image.fromarray(pixel).convert('L')
            image_name = os.path.join(subfolder, '{:05d}.jpg'.format(i))
            print(image_name)
            im.save(image_name)

# 保存模型权重
emotion_model.save_weights(r'C:\Apps\Anconda_new\dataset\fer2013\model.h5')
# 不知道为什么文件保存错误

# 载入模型权重
# model.load_weights('model.h5')

# 使用摄像头并预测情绪

cv2.ocl.setUseOpenCL(False)

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Sad", 5: "Surprised", 6: "Neutral"}

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    bounding_box = cv2.CascadeClassifier('C:\\Apps\\anaconda\\pkgs\\libopencv-3.4.2-h20b85fd_0\\Library\\etc\\haarcascades\\haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    num_faces = bounding_box.detectMultiScale(gray_frame,scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
        emotion_prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
        cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('Video', cv2.resize(frame,(1200,860),interpolation = cv2.INTER_CUBIC))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
    break

# GUI

import tkinter as tk
from tkinter import *
import cv2
from PIL import Image, ImageTk
import os
import numpy as np
import cv2

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from keras.layers import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator

emotion_model = Sequential()

emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))

emotion_model.load_weights(r'C:\Apps\Anconda_new\dataset\fer2013\model.h5', by_name=False)
# ,by_name=False

# if swmr and swmr_support:
#     flags |= h5f.ACC_SWMR_READ
#     fid = h5f.open(model.h5, flags, fapl=fapl)
# elif mode == 'r+':
#     fid = h5f.open(name, h5f.ACC_RDWR, fapl=fapl)
# emotion_model.load_weights('C:\\Apps\\Anconda_new\\dataset\\model.h5')# 载入模型权重

cv2.ocl.setUseOpenCL(False)

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Sad", 5: "Surprised", 6: "Neutral"}
emoji_dist = {0: r"C:\Apps\Anconda_new\dataset\fer2013\emojis\angry.png",
              2: r"C:\Apps\Anconda_new\dataset\fer2013\emojis\disgusted.png",
              2: r"C:\Apps\Anconda_new\dataset\fer2013\emojis\fearful.png",
              3: r"C:\Apps\Anconda_new\dataset\fer2013\emojis\happy.png",
              4: r"C:\Apps\Anconda_new\dataset\fer2013\emojis\sad.png",
              5: r"C:\Apps\Anconda_new\dataset\fer2013\emojis\surpriced.png",
              6: r"C:\Apps\Anconda_new\dataset\fer2013\emojis\neutral.png"}
global last_frame1
last_frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
global cap1
cap1 = cv2.VideoCapture(0)
show_text = [0]


def show_vid():
    if not cap1.isOpened():
        print("cant open the camera1")
    flag1, frame1 = cap1.read()
    frame1 = cv2.resize(frame1, (600, 500))

    bounding_box = cv2.CascadeClassifier(
        'C:\\Apps\\anaconda\\pkgs\\libopencv-3.4.2-h20b85fd_0\\Library\\etc\\haarcascades\\haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    num_faces = bounding_box.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame1, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
        prediction = emotion_model.predict(cropped_img)

        maxindex = int(np.argmax(prediction))
        cv2.putText(frame1, emotion_dict[maxindex], (x + 20, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                    cv2.LINE_AA)
        show_text[0] = maxindex
    if flag1 is None:
        print("Major error!")
    elif flag1:
        global last_frame1
        last_frame1 = frame1.copy()
        pic = cv2.cvtColor(last_frame1, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(pic)
        imgtk = ImageTk.PhotoImage(image=img)
        lmain.imgtk = imgtk
        lmain.configure(image=imgtk)
        lmain.after(10, show_vid)


def show_vid2():
    print('vid2')
    frame2 = cv2.imread(emoji_dist[show_text[0]])
    pic2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
    img2 = Image.fromarray(frame2)
    imgtk2 = ImageTk.PhotoImage(image=img2)
    lmain2.imgtk2 = imgtk2
    lmain3.configure(text=emotion_dict[show_text[0]], font=('arial', 45, 'bold'))

    lmain2.configure(image=imgtk2)
    lmain2.after(10, show_vid2)


if __name__ == '__main__':
    root = tk.Tk()
    # 感觉这个code没什么用，而且在fer2013中没有logo文件，所以就注释掉了
    #     img = ImageTk.PhotoImage(Image.open("C:\\Apps\\Anconda_new\\dataset\\fer2013\\logo\"))
    #     heading = Label(root,image=img,bg='black')

    #     heading.pack()
    heading2 = Label(root, text="Photo to Emoji", pady=20, font=('arial', 45, 'bold'), bg='black', fg='#CDCDCD')

    heading2.pack()
    lmain = tk.Label(master=root, padx=50, bd=10)
    lmain2 = tk.Label(master=root, bd=10)
    lmain3 = tk.Label(master=root, bd=10, fg="#CDCDCD", bg='black')
    lmain.pack(side=LEFT)
    lmain.place(x=50, y=250)
    lmain3.pack()
    lmain3.place(x=960, y=250)
    lmain2.pack(side=RIGHT)
    lmain2.place(x=900, y=350)

    root.title("Photo To Emoji")
    root.geometry("1400x900+100+10")  # x是小写字母X不是乘号
    root['bg'] = 'black'
    exitbutton = Button(root, text='Quit', fg="red", command=root.destroy, font=('arial', 25, 'bold')).pack(side=BOTTOM)

    show_vid()
    show_vid2()
    root.mainloop()
    cap1.release()

