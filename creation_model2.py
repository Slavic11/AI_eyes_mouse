import matplotlib
matplotlib.use("Agg")
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers.core import Dense
from tensorflow.keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import random
import pickle
import cv2
import os
from sklearn.metrics import mean_squared_error

from keras.models import load_model

from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from tensorflow import keras

from keras.utils.vis_utils import plot_model

from keras.layers import concatenate
from keras.layers import Input
from keras.models import Model
from keras.layers import Conv2D, Flatten, Dropout, Activation, MaxPooling2D, AveragePooling2D 
from tensorflow.keras.optimizers import Adam





ImagePaths = list(paths.list_images("C:/Users/Admin/Desktop/Pro_dip/Eye_obsch_MS"))
print(len(ImagePaths))
random.shuffle(ImagePaths)

data = []
labels = []
bonuses = []

for imagepath in ImagePaths:
  
  image = cv2.imread(imagepath)
  if image is None:  
    continue  




  image = cv2.resize(image, (200,50))
  image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  data.append(image)
  #print(image)
  label = imagepath.split(os.path.sep)[-1]

  distan_ce = int(label.split('.')[0].split('_')[1])

  if int(label.split('.')[0].split('_')[2]) != 0 :
    x_centre = int(label.split('.')[0].split('_')[2])/1280 
  else:
    x_centre = 0
  if int(label.split('.')[0].split('_')[3]) != 0:
    y_centre = int(label.split('.')[0].split('_')[3])/800
  else:
    y_centre = 0

  position_x = int(label.split('.')[0].split('_')[4])
  position_z = int(label.split('.')[0].split('_')[5])
  

  if int(label.split('.')[0].split('_')[6]) != 0 :
    x_norm = int(label.split('.')[0].split('_')[6])/1920
  else:
    x_norm = 0
  if int(label.split('.')[0].split('_')[7]) != 0:
    y_norm = int(label.split('.')[0].split('_')[7])/1080
  else:
    y_norm = 0
  

  label = ([x_norm, y_norm])
  bonus = ([distan_ce, position_x, position_z])

  labels.append(label)
  bonuses.append(bonus)



data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)
bonuses = np.array(bonuses)
print(data[1])


(trainX1, testX1, trainX2, testX2, trainY, testY) = train_test_split(data, bonuses, labels, test_size=0.1, random_state =43)




with open("tipamodeltest/pickle/trainX1.pickle", 'wb') as f:         #here and below saving and loading pickle files with datasets
  pickle.dump(trainX1, f)                                            #It's simple, so as not to start training again every time - you need to save all the data before training
print ("trainX1 saved")                                              #before starting re-training, you need to change pickle files from saving to loading.

with open("tipamodeltest/pickle/testX1.pickle", 'wb') as f:
  pickle.dump(testX1, f)
print ("testX1 saved")

with open("tipamodeltest/pickle/trainX2.pickle", 'wb') as f:
  pickle.dump(trainX2, f)
print ("trainX2 saved")

with open("tipamodeltest/pickle/testX2.pickle", 'wb') as f:
  pickle.dump(testX2, f)
print ("testX2 saved")

with open("tipamodeltest/pickle/trainY.pickle", 'wb') as f:
  pickle.dump(trainY, f)
print ("trainY saved")

with open("tipamodeltest/pickle/testY.pickle", 'wb') as f:
  pickle.dump(testY, f)
print ("testY saved")

"""
with open("tipamodeltest/pickle/trainX1.pickle", 'rb') as f:
  pickle.load(f) 
print ("trainX1 loaded")

with open("tipamodeltest/pickle/testX1.pickle", 'rb') as f:
  pickle.load(f) 
print ("testX1 loaded")

with open("tipamodeltest/pickle/trainX2.pickle", 'rb') as f:
  pickle.load(f) 
print ("trainX2 loaded")

with open("tipamodeltest/pickle/testX2.pickle", 'rb') as f:
  pickle.load(f) 
print ("testX2 loaded")

with open("tipamodeltest/pickle/trainY.pickle", 'rb') as f:
  pickle.load(f) 
print ("trainY loaded")

with open("tipamodeltest/pickle/testY.pickle", 'rb') as f:
  pickle.load(f) 
print ("testY loaded")
"""


#############################################################################
#############################################################################
#############################################################################
#############################################################################
"""
inputsimg = keras.Input(shape=(50,200,1), name='img_input')
inputsbon = keras.Input(shape=(3), name='ts_input')


x = Conv2D(16,(7,7), padding = "valid", activation="relu")(inputsimg)            
x = Conv2D(40,(7,7), padding = "same", activation="relu")(x)

x = MaxPooling2D(pool_size=(2, 2))(x)                                         
x = Dropout(0.25)(x)



x = Flatten()(x)

combined = concatenate([x, inputsbon])
z = Dense(2, activation="tanh")(combined)

model = keras.Model(inputs=[inputsimg, inputsbon], outputs=z)
"""
#############################################################################
#############################################################################
#############################################################################
#############################################################################


model = load_model("tipamodeltest/point63.h5")                                    #continued learning   tipamodeltest/point99.h5
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

opt = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
   
model.compile(loss="mean_squared_error",optimizer=opt, metrics=["accuracy"]) 

print ("Model compiled")
model.summary()


EPOCHS = 200

checkpointer = ModelCheckpoint(filepath='C:/Users/Admin/Desktop/Pro_dip/tipamodeltest/point.h5', verbose=1, save_best_only=True)


               


H = model.fit({'img_input':trainX1, 'ts_input':trainX2}, trainY, validation_data=([testX1, testX2], testY), 
  epochs=EPOCHS, batch_size=100,   
  shuffle=True, callbacks=[checkpointer])       
print ("Model trained")









predictions = model.predict ((testX1,testX2),batch_size = 32)

print(classification_report(testY.argmax(axis=1),
  predictions.argmax(axis=1), target_names=("X","Y")))

N = np.arange(0,EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="vall_loss")
plt.plot(N, H.history["accuracy"], label="train_acc")
plt.plot(N, H.history["val_accuracy"], label="vall_acc")
plt.title("Results")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy/")
plt.legend()
plt.savefig("C:/Users/Admin/Desktop/Pro_dip/tipamodeltest/Loss.png")

model.save("C:/Users/Admin/Desktop/Pro_dip/tipamodeltest/EsyNet.model")

print ("End")











print("ok")
