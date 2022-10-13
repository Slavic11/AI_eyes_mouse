import cv2
import numpy as np
import dlib
import math
import keyboard
import pyautogui


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

key = 1
ESCAPE = 27
imgnum = 0


mam_px = 148
slav_px = 148
cam_chin_sm = 50

fx = 1980
fy = 1080

cap = cv2.VideoCapture(0)

cap.set(3,fx) 
cap.set(4,fy)

while (key != ESCAPE):
  ret, frame = cap.read()
  width, height = pyautogui.position()
  faceszz = detector(frame)
  for face in faceszz:
    landmarks = predictor(frame, face)
    for n in range(0, 68): 
      x1 = landmarks.part(37).x
      y1 = landmarks.part(37).y   
      x2 = landmarks.part(46).x
      y2 = landmarks.part(46).y
      nose = landmarks.part(30)
      chin = landmarks.part(8)    
      left_eye = landmarks.part(17)
      right_eye = landmarks.part(26)
      mouth_left = landmarks.part(48)
      mouth_right  = landmarks.part(54)


      cv2.circle(frame, (x1, y1), 3, (255, 0, 0), -1)
      cv2.circle(frame, (x2, y2), 3, (255, 0, 0), -1)


      
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




    chin_nos_pix = chin.y - nose.y
    fk_sm = round((cam_chin_sm * slav_px) / chin_nos_pix) 

    cv2.putText(frame,"sm=" + str(round(fk_sm, 1)) + " 0=" + str(nose) + "X=" + str(X) + " Y=" + str(Y) + " Z=" + str(Z), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    if(keyboard.is_pressed('space')):
        imgnum += 1
        print ("x=" + str(width) + " y=" + str(height))
        screenshot = pyautogui.screenshot(region=(x1-32, y1+19, (x2+40)-(x1-40), (y2+20)-(y1-20)))
        screenshot.save("Eye_test/eye2S" + str(imgnum) + "_" + str(fk_sm) + "_" + str(nose.x) + "_" + str(nose.y) + "_" + 
            str(X) + "_" + str(Z) + "_" + str(width) + "_" + str(height) + ".png")

    cv2.imshow('frame', frame)
    key = cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
