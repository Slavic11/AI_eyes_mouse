from keras.models import load_model
from imutils import  paths
import cv2
import numpy as np


model = load_model("tipamodeltest/point.h5")  

x_ob=0
y_ob=0

ImagePaths = list(paths.list_images("Eye_test"))  
for imagepath in ImagePaths:                  
  data = []    
  bonuses = []                                  
  image = cv2.imread(imagepath)
  print(imagepath)

  distan_ce = int(imagepath.split('.')[0].split('_')[2])
  position_x = int(imagepath.split('.')[0].split('_')[5])
  position_z = int(imagepath.split('.')[0].split('_')[6])
  bonus = ([distan_ce, position_x, position_z])


  x=int(imagepath.split('.')[0].split('_')[7]) 
  y=int(imagepath.split('.')[0].split('_')[8]) 
  print("imagepath" + " x=" + str(x) + " y=" + str(y))


  image = cv2.resize(image, (200, 50))

  image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)         

  data.append(image)
  bonuses.append(bonus)

  data = np.array(data, dtype="float") / 255.0  
  bonuses = np.array(bonuses)

  pred = model.predict([[data, bonuses]])
  pred[0,0] = pred[0,0] * 1920
  pred[0,1] = pred[0,1] * 1080
  print(pred) 

  xrazn = round((pred[0,0]) - x)
  yrazn = round((pred[0,1]) - y)
  print("pred " + "разница X= " + str(xrazn) + " разница Y= " + str(yrazn))
  
  if(xrazn > 0):
    x_ob += xrazn
  else:
    x_ob += xrazn * (-1)
  if(yrazn > 0):
    y_ob += yrazn
  else:
    y_ob += yrazn * (-1)


  print("\n")

print("x= " + str(x_ob/24));
print("y= " + str(y_ob/24));

