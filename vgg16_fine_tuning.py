# Modules #########################################################

import os
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Input, Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
import numpy as np

# Parameters #######################################################
x_train,x_test,y_train_,y_test = np.load(r"C:\CMD\Tensorflow\Fine_Tuning\train\data.npy")
f_size = 200
train_data_dir = r'C:\CMD\Tensorflow\Fine_Tuning\train'
validation_data_dir = r'C:\CMD\Tensorflow\Fine_Tuning\validation'
samples_per_epoch = 10
nb_val_samples = 100
nb_epoch = 50
result_dir = r'C:\CMD\Tensorflow\Fine_Tuning\results'

if __name__ == '__main__':
    # VGG16モデルと学習済み重みをロード
    # Fully-connected層（FC）はいらないのでinclude_top=False）
    # input_tensorを指定しておかないとoutput_shapeがNoneになってエラーになるので注意
    # https://keras.io/applications/#inceptionv3
    input_tensor = Input(shape=(f_size,f_size, 3))
    vgg16_model = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)
    # vgg16_model.summary()

    # FC層を構築
    # Flattenへの入力指定はバッチ数を除く
    top_model = Sequential()
    top_model.add(Flatten(input_shape=vgg16_model.output_shape[1:]))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(1, activation='sigmoid'))

    # vgg16_modelはkeras.engine.training.Model
    # top_modelはSequentialとなっている
    # ModelはSequentialでないためadd()がない
    # そのためFunctional APIで二つのモデルを結合する
    # https://github.com/fchollet/keras/issues/4040
    model = Model(input=vgg16_model.input, output=top_model(vgg16_model.output))
    print('vgg16_model:', vgg16_model)
    print('top_model:', top_model)
    print('model:', model)

    # layerを表示
    for i in range(len(model.layers)):
        print(i, model.layers[i])

    # 最後のconv層の直前までの層をfreeze
    for layer in model.layers[:15]:
        layer.trainable = False

    # TODO: ここでAdamを使うとうまくいかない
    # Fine-tuningのときは学習率を小さくしたSGDの方がよい？
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                  metrics=['accuracy'])

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(f_size,f_size),
        batch_size=32,
        class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(f_size,f_size),
        batch_size=32,
        class_mode='binary')
    # fitを使う場合
    history = model.fit(x_train, y_train,
                    batch_size=10,
                    epochs=2,
                    verbose=1,
                    validation_data=(x_test, y_test))

    # Fine-tuning
    history = model.fit_generator(
        train_generator,
        samples_per_epoch=samples_per_epoch,
        nb_epoch=nb_epoch,
        validation_data=validation_generator,
        nb_val_samples=nb_val_samples)

    model.save_weights(os.path.join(result_dir, 'finetuning.h5'))
