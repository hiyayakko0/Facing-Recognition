# Modules #########################################################
from __future__ import print_function

import os
import numpy as np
from argparse import ArgumentParser

# base models
from keras.applications.xception import Xception
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16

from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator,load_img, img_to_array
from keras.utils import np_utils
from keras.callbacks import CSVLogger, ModelCheckpoint
from sklearn import model_selection

# Parameter ##############################################################
# 作成するデータセットパス
# データセットパス以下に入れたフォルダがそのままカテゴリになる。
# 00Hanako
# 01Taro
#　のようにつける

# Main ###################################################################
def main():
    dataset_path = r"E:\seishi170\Documents\YANAI\2018.02.13_Face_Recognition_test\Fine_Tuning"
    # parse args
    options = Parser().parse_args()
    image_size = int(options.image_size)
    if not dataset_path:
        dataset_path = options.dataset_path
    output_path = os.path.join(dataset_path,options.output_path)
    batch_size_pre = int(options.batch_size_pre)
    batch_size_fine = int(options.batch_size_fine)
    nb_epoch_pre = int(options.nb_epoch_pre)
    nb_epoch_fine = int(options.nb_epoch_fine)

    device_id = int(options.device_id)
    train_name = 'set' + '_' + str(image_size)
    base_model_name = options.base_model_name

    print('image size                      :', image_size)
    print('dataset path                    :', dataset_path)
    print('batch size (pre-training)       :', batch_size_pre)
    print('batch size (fine-tuning)        :', batch_size_fine)
    print('number of epochs (pre-training) :', nb_epoch_pre)
    print('number of epochs (fine-tuning)  :', nb_epoch_fine)
    print('output path                     :', output_path)
    print('gpu device id                   :', device_id)
    print('train name                      :', train_name)
    print('model                           :', base_model_name)

    # set gpu device
#    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    
    # The data, shuffled and split between train and test sets:
    loader = DataLoader(image_size,dataset_path)
    (X_train, y_train), (X_test, y_test),nb_classes = loader.load_data()
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # Convert class vectors to binary class matrices.
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    X_test = X_test.astype('float32')
    X_test /= 255



    ### add for TensorBoard ##############
    import keras.callbacks
    import keras.backend.tensorflow_backend as KTF
    import tensorflow as tf
    
    old_session = KTF.get_session()
    
    session = tf.Session('')
    KTF.set_session(session)
    KTF.set_learning_phase(1)
    #######################################

    # create the base pre-trained model
    base_model = VGG16(weights='imagenet', include_top=False)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer and a logistic layer -- let's say we have 25 classes
    predictions = Dense(nb_classes, activation='softmax')(x)

    # this is the model we will train
    model = Model(input=base_model.input, output=predictions)
    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional Xception layers
    # len(base_model.layers)
    # model.summary()
    for layer in base_model.layers[:17]:
        layer.trainable = False
    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(
            optimizer='rmsprop', 
            loss='categorical_crossentropy', 
            metrics=['accuracy']
                  )


    ### add for TensorBoard ################################
    tb_cb = keras.callbacks.TensorBoard(log_dir=output_path+"\\log\\", histogram_freq=1)
    cbks = [tb_cb]
    #######################################################

    # callbacks
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    logger_pre = CSVLogger(
            output_path + '/' + train_name + '_training_pre.csv'
            )
    logger_fine = CSVLogger(
            output_path + '/' + train_name + '_training_fine.csv'
            )
    checkpointer = ModelCheckpoint(
            filepath=output_path + '/' + train_name + '.hdf5', 
            verbose=1, 
            save_best_only=True
            )
    
    ##################################################################################################
    # pre-training
    ##################################################################################################
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = createImageDataGenerator()

    # Compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(X_train)

    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size_pre),
                        samples_per_epoch=X_train.shape[0],
                        nb_epoch=nb_epoch_pre,
                        callbacks=[logger_pre, checkpointer],
                        validation_data=(X_test, Y_test))
    
#    model.save(output_path+"\\model.h5")
    ### add for TensorBoard ##################
    KTF.set_session(old_session)
    ##########################################

    ########################################################################
    # fine-tuning
    ########################################################################
#    # we chose to train all layers
#    for layer in model.layers:
#       layer.trainable = True
#
#    # we need to recompile the model for these modifications to take effect
#    # we use SGD with a low learning rate
#    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
#
#    # we train our model again (this time fine-tuning all layers)
#    print('Using real-time data augmentation.')
#    # This will do preprocessing and realtime data augmentation:
#    datagen = createImageDataGenerator()
#
#    # Compute quantities required for featurewise normalization
#    # (std, mean, and principal components if ZCA whitening is applied).
#    datagen.fit(X_train)
#
#    # Fit the model on the batches generated by datagen.flow().
#    model.fit_generator(
#            datagen.flow(X_train, Y_train, batch_size=batch_size_fine),
#            samples_per_epoch=X_train.shape[0],
#            nb_epoch=nb_epoch_fine,
#            callbacks=[logger_fine, checkpointer],
#            validation_data=(X_test, Y_test)
#                        )



# Functions ############################################################## 
def Parser():
    parser = ArgumentParser()
    parser.add_argument('--image_size', default=299)
    parser.add_argument('--dataset_path', default=r".\\")
    parser.add_argument('--batch_size_pre', default=32)#32
    parser.add_argument('--batch_size_fine', default=16)#16
    parser.add_argument('--nb_epoch_pre', default=5)#5
    parser.add_argument('--nb_epoch_fine', default=100)
    parser.add_argument('--output_path', default='output')
    parser.add_argument('--device_id', default=0)
    parser.add_argument('--base_model_name' , default='Xception')
    return parser

class DataLoader():
    def __init__(self, image_size,dataset_path):
        self.image_size = image_size
        self.dataset_path = dataset_path

    def load_data(self):
        categories = [
                   fls 
                   for fls in os.listdir(os.path.join(self.dataset_path,"data"))
                   if os.path.isdir(os.path.join(self.dataset_path,"data",fls))
                    ]
            
        nb_classes = len(categories)
        
        datas =[
                [os.path.join(self.dataset_path+"\\data",name,j),i]
                for i,name in enumerate(categories)
                for j in os.listdir(os.path.join(self.dataset_path+"\\data",name))
                ]
        X = []
        Y = []
        for data in datas:
                img = load_img(
                        data[0], 
                        target_size=(self.image_size,self.image_size)
                               )
                img = img_to_array(img)
                X.append(img)
                Y.append(data[1])
                
        X = np.array(X)
        Y = np.array(Y)
        
        X_train, X_test, y_train, y_test = \
            model_selection.train_test_split(X, Y)
        # データを保存する時
#        xy = (X_train, X_test, y_train, y_test)
#        np.save(os.path.join(self.dataset_path,"data.npy"), xy)
        print("Data loaded:", len(Y))

        return (X_train,y_train), (X_test,y_test), nb_classes

def createImageDataGenerator():
    return ImageDataGenerator(
        rescale=1./255,
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=5,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.10,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.10,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        zoom_range=0.5,
        vertical_flip=False)  # randomly flip images

if __name__ == "__main__":
    main()