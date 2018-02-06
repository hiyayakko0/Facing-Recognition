# -*- coding: utf-8 -*-
"""
参考：
[openCVで複数画像ファイルから顔検出をして切り出し保存]
(https://qiita.com/FukuharaYohei/items/457737530264572f5a5b)
指定フォルダ以下すべての写真フォルダの画像を取得
保存する顔画像サイズの拡大
ファイルの移動をデフォルトでOFF

"""
import cv2, os, argparse, shutil

# parameter ###################################
# 切り抜いた画像の保存先ディレクトリ
pic_path = r"C:\CMD\OpenCV\TEST"
save_path = r"C:\CMD\OpenCV\TEST"
#cascade_path = r"C:\CMD\Anaconda\Library\etc\haarcascades"
cascade_path = r"C:\Anaconda3\Library\etc\haarcascades"
SAVE_PATH+"\\output"
# functions ########################################
os.chdir(pic_path)
# 学習済モデルの種類
CASCADE = ["default","alt","alt2","tree","profile","nose"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser() #パーサをコンストラクト
    parser.add_argument(
        "--cascade",
        type=str,
        default="alt",
        choices=CASCADE,
        help="cascade file."
  )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.3,
        help="scaleFactor value of detectMultiScale."
  )
    #　顔通しの最小距離
    parser.add_argument(
        "--neighbors",
        type=int,
        default=2,
        help="minNeighbors value of detectMultiScale."
  )
    # 顔として認識する最小サイズ
    parser.add_argument(
        "--min",
        type=int,
        default=50,
        help="minSize value of detectMultiScale."
  )
    # インプットディレクトリ
    parser.add_argument(
        "--input_dir",
        type=str,
        default="./",
        help="The path of input directory."
  )
    # 検出済みファイルの移動ディレクトリ
    # 指定しない場合は移動しない
    parser.add_argument(
        "--move_dir",
        type=str,
        default="./",
        help="The path of moving detected files."
  ) 
    # 検出された顔サイズに対して保存するどれだけ大きくするか
    parser.add_argument(
        "--face_size",
        type=float,
        default=1.1,
        help="Face size."
  )

# パラメータ取得と実行
FLAGS, unparsed = parser.parse_known_args()

# 分類器ディレクトリ(以下から取得)
# https://github.com/opencv/opencv/blob/master/data/haarcascades/
# https://github.com/opencv/opencv_contrib/blob/master/modules/face/data/cascades/

# 学習済モデルファイル
if   FLAGS.cascade == CASCADE[0]:#"default":
    cascade_path = os.path.join(cascade_path,"haarcascade_frontalface_default.xml")
elif FLAGS.cascade == CASCADE[1]:#"alt":
    cascade_path = os.path.join(cascade_path,"haarcascade_frontalface_alt.xml")
elif FLAGS.cascade == CASCADE[2]:#"alt2":
    cascade_path = os.path.join(cascade_path,"haarcascade_frontalface_alt2.xml")
elif FLAGS.cascade == CASCADE[3]:#"tree":
    cascade_path = os.path.join(cascade_path,"haarcascade_frontalface_alt_tree.xml")
elif FLAGS.cascade == CASCADE[4]:#"profile":
    cascade_path = os.path.join(cascade_path,"haarcascade_profileface.xml")
elif FLAGS.cascade == CASCADE[5]:#"nose":
    cascade_path = os.path.join(cascade_path,"haarcascade_mcs_nose.xml")

#カスケード分類器の特徴量を取得する
faceCascade = cv2.CascadeClassifier(cascade_path)

# 顔検知に成功した数(デフォルトで0を指定)
face_detect_count = 0

# 顔検知に失敗した数(デフォルトで0を指定)
face_undetected_count = 0

# フォルダ内ファイルを変数に格納(ディレクトリも格納)
rt = os.walk(FLAGS.input_dir)
files = list()
for fls in rt:
    for fl in fls[2]:
        files.append(os.path.join(fls[0],fl))

if not os.path.exists(save_path+"\\output"):
    os.mkdir(save_path+"\\output")

if not os.path.exists(FLAGS.move_dir):
    os.mkdir(FLAGS.move_dir)


print(FLAGS)

# 集めた画像データから顔が検知されたら、切り取り、保存する。
for file_name in files:

    # ファイルの場合(ディレクトリではない場合)
    if os.path.isfile(os.path.join(FLAGS.input_dir,file_name)):

        # 画像ファイル読込
        img = cv2.imread(os.path.abspath(file_name))

        # 大量に画像があると稀に失敗するファイルがあるのでログ出力してスキップ(原因不明)
        if img is None:
            print(file_name + ':Cannot read image file')
            continue

        # カラーからグレースケールへ変換(カラーで顔検出しないため)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 顔検出
        face = faceCascade.detectMultiScale(gray,
                                            scaleFactor=FLAGS.scale,
                                            minNeighbors=FLAGS.neighbors, 
                                            minSize=(FLAGS.min, FLAGS.min)
                                            )
        if len(face) > 0:
            for (x,y,w,h) in face:
                # 切り取った画像出力
                rt = FLAGS.face_size-1
                size = max(w,h)
                x = max(x-int(w*rt),0)
                y = max(y-int(h*rt),0)
                
                cv2.imwrite(save_path+"\\output\\"+str(face_detect_count)+os.path.basename(file_name), 
                            img[y:y+int(size*(rt+1)),#上から下へ 
                                x:x+int(size*(rt+1))]#右から左へ
                            )
                face_detect_count = face_detect_count + 1

            # 検出できたファイルは移動
            if FLAGS.move_dir != ("" or "./"):
                shutil.move(FLAGS.input_dir + file_name, 
                            FLAGS.move_dir)
        else:
            print(file_name + ':No Face')
            face_undetected_count = face_undetected_count + 1
            
print('Undetected Image Files:%d' % face_undetected_count)
