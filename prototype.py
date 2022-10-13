import cv2
import numpy as np
import dlib
import time
import math
import keyboard
import pyautogui

from keras.models import load_model

from playsound import playsound        


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

model = load_model("tipamodeltest/point90.h5")       

key = 1
ESCAPE = 27
imgnum = 0

button = 'left'             
click = 1

mam_px = 148
slav_px = 148
cam_chin_sm = 50

cap = cv2.VideoCapture(0)

fx = 1980
fy = 1080

cap.set(3,fx) 
cap.set(4,fy)

while (key != ESCAPE):
  ret, frame = cap.read()
  width, height = pyautogui.position()
  faceszz = detector(frame)
  for face in faceszz:
    landmarks = predictor(frame, face)
    for n in range(0, 68): 
      x1 = landmarks.part(37) 
      x2 = landmarks.part(46)

      lips_top = landmarks.part(62)
      lips_bottom = landmarks.part(66)

      nose = landmarks.part(30)
      chin = landmarks.part(8)    
      left_eye = landmarks.part(17)
      right_eye = landmarks.part(26)
      mouth_left = landmarks.part(48)
      mouth_right  = landmarks.part(54)


      


      cv2.circle(frame, (nose.x, nose.y), 3, (255, 0, 0), -1)
      cv2.circle(frame, (chin.x, chin.y), 3, (255, 0, 0), -1)
      cv2.circle(frame, (mouth_left.x, mouth_left.y), 3, (255, 0, 0), -1)
      cv2.circle(frame, (mouth_right.x, mouth_right.y), 3, (255, 0, 0), -1)
      

    image_points = np.array([
            (nose.x, nose.y),  
            (chin.x, chin.y),  
            (left_eye.x, left_eye.y),  
            (right_eye.x, right_eye.y),  
            (mouth_left.x, mouth_left.y),  
            (mouth_right.x, mouth_right.y)  
            ], dtype="double")
    

    model_points = np.array([
             (0.0, 0.0, 0.0), 
             (0.0, -330.0, -65.0), 
             (-225.0, 170.0, -135.0),
             (225.0, 170.0, -135.0),
             (-150.0, -150.0, -125.0), 
             (150.0, -150.0, -125.0) 
             ])

    
    
    center = (fx / 2, fy / 2) 
    cx = center[0]
    cy = center[1]

    camera_matrix = np.array(
    [[fx, 0, cx],
    [0, fx, cy],
    [0, 0, 1]], 
    dtype=np.float64)

    
    dist_coeffs = np.zeros ((4, 1)) 
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)



    
    theta = cv2.norm(rotation_vector, cv2.NORM_L2)


    w = math.cos(theta / 2)
    x = math.sin(theta / 2) * rotation_vector[0][0] / theta
    y = math.sin(theta / 2) * rotation_vector[1][0] / theta
    z = math.sin(theta / 2) * rotation_vector[2][0] / theta

    ysqr = y * y
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + ysqr)
    pitch = math.atan2(t0, t1)


    t2 = 2.0 * (w * y - z * x)
    if t2 > 1.0:
        t2 = 1.0
    if t2 < -1.0:
        t2 = -1.0
    yaw = math.asin(t2)

    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (ysqr + z * z)
    roll = math.atan2(t3, t4)

    Y = int((pitch / math.pi) * 180)
    X = int((yaw / math.pi) * 180)
    Z = int((roll / math.pi) * 180)   


    euler_angle_str = 'X:{}, Y:{}, Z:{}'.format(pitch, yaw, roll)






    frame_re = frame[x1.y-20              :x1.y-20    +(x2.y-x1.y)+40,                  
    x1.x-35:            x1.x-35    +(x2.x+40)-(x1.x-40)] 



    chin_nos_pix = chin.y - nose.y
    fk_sm = round((cam_chin_sm * slav_px) / chin_nos_pix)

    cv2.putText(frame,"sm=" + str(round(fk_sm, 1)) + " 0=" + str(nose) + "X=" + str(X) + " Y=" + str(Y) + " Z=" + str(Z), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    data = []
    bonuses = []
    imagepatth = pyautogui.screenshot(region=(x1.x-32, x1.y+19, (x2.x+40)-(x1.x-40), (x2.y+20)-(x1.y-20)))
    image = imagepatth


    imagepatth = np.array(imagepatth)   
    imagepatth = cv2.resize(imagepatth, (200, 50))  
    imagepatth = cv2.cvtColor(imagepatth, cv2.COLOR_BGR2GRAY)
    data.append(imagepatth) 


    
    data = np.array(data, dtype = "float") / 255.0    

    distan_ce = int(fk_sm)
    position_x = int(X)
    position_z = int(Z)
    bonus = ([distan_ce, position_x, position_z])
    bonuses.append(bonus)
    bonuses = np.array(bonuses)

    if((lips_bottom.y - lips_top.y) > 15):
      pyautogui.click(pred[0,0], pred[0,1], click, 0.05, button)                   

    pred = model.predict([[data, bonuses]]) 
    pred[0,0] = pred[0,0] * 1920
    pred[0,1] = pred[0,1] * 1080
    print(lips_bottom.y - lips_top.y)                      
    print(pred)
    print("\n")
    time.sleep(0.2)
    if (pred[0,0] < 0):
      pred[0,0] = 1

    if(position_x >= 18 and position_x < 25):
      playsound("click_01.mp3")              
      button = 'right'
      click = 1
      time.sleep(0.2)
    if(position_x >= 25):
      playsound("double_click_01.mp3")              
      button = 'right'
      click = 2
      time.sleep(0.2)

    if(position_x > -25 and position_x < -18):
      playsound("click_01.mp3")               
      button = 'left'
      click = 1
      time.sleep(0.2)
    if(position_x <= -25):
      playsound("double_click_01.mp3")              
      button = 'left'
      click = 2
      time.sleep(0.2)

    pyautogui.moveTo(pred[0,0], pred[0,1], duration=0.1)   

    

 
    cv2.imshow('frame', frame)
    cv2.setWindowProperty('frame', cv2.WND_PROP_TOPMOST, 1)     
    key = cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()