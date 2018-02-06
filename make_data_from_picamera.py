# -*- coding: utf-8 -*-
"""
定期的にpicameraから取得した画像から顔を切り出す
定期的にpicameraからすべての画像も保存する
"""
import picamera
import cv2, os, argparse, shutil
import datetime
import time
# parameter ###################################
# 切り抜いた画像の保存先ディレクトリ
save_path = r"C:\CMD\OpenCV\TEST"
#cascade_path = r"C:\CMD\Anaconda\Library\etc\haarcascades"
cascade_path = r"C:\Anaconda3\Library\etc\haarcascades\haarcascade_frontalface_alt.xml"
# functions ########################################
os.chdir(save_path)
if not os.path.exists("./face"):
    os.mkdir("./face")
if not os.path.exists("./pic"):
    os.mkdir("./pic")

f_folder = os.path.join(save_path,"face")
p_folder = os.path.join(save_path,"pic")

faceCascade = cv2.CascadeClassifier(cascade_path)

# 顔検知に成功した数(デフォルトで0を指定)
face_detect_count = 0

# 顔検知に失敗した数(デフォルトで0を指定)
pic_count = 0

# フォルダ内ファイルを変数に格納(ディレクトリも格納)
while True:
    now = datetime.datetime.now()
    pic_count += 1
    with picamera.PiCamera() as camera:
            # Photo dimensions and rotation
            time.sleep(3)  # カメラ初期化
            if pic_count % 5 == 0:
                pf = os.path.join(p_folder,'pic_{0:%Y%m%d%H%M%S}.jpg'.format(now))
                camera.capture(pf)
    img = cv2.imread(???)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = faceCascade.detectMultiScale(gray,
                                        scaleFactor=1.3,
                                        minNeighbors=2, 
                                        minSize=(50,50)
                                        )
    if len(face) > 0:
        for (x,y,w,h) in face:
            # 切り取った画像出力
            rt = 0.1
            size = max(w,h)
            x = max(x-int(w*rt),0)
            y = max(y-int(h*rt),0)

            # 顔写真の保存
            ff = os.path.join(f_folder,'face_{0:%Y%m%d%H%M%S}.jpg'.format(now))
            cv2.imwrite(ff, 
                        img[y:y+int(size*(rt+1)),#上から下へ 
                            x:x+int(size*(rt+1))]#右から左へ
                        )
            face_detect_count = face_detect_count + 1
    if face_detect_count >= 5:
        break





