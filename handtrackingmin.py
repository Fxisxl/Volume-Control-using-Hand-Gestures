import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands

#below obj only accepts RGB img

hands = mpHands.Hands()
#for drawing points & connections
mpDraw = mp.solutions.drawing_utils

#to display fps

prevTime= 0
currTime= 0


while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(imgRGB)
    #print(result.multi_hand_landmarks)

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks: #here handLms is single hand or hand(0)

            for id, lm in enumerate(handLms.landmark):
                 #height width breadth
                 h, w, c = img.shape

                 #to get the pixel co-ordinate on our img (ie vid)
                 cx, cy = int(lm.x * w), int (lm.y * h)  #here lm.x or y are the coordinate given by tracking in decimalform

                 #print(id, cx, cy)
                 if id==None:
                   cv2.circle(img,(cx,cy), 15, (255, 0, 255), cv2.FILLED)



            mpDraw.draw_landmarks(img , handLms,mpHands.HAND_CONNECTIONS)

    #code for fps
    currTime= time.time()
    fps=1/(currTime-prevTime)
    prevTime=currTime

    cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_ITALIC,2 ,(255,0,255) ,3 )


    cv2.imshow("Hand", img)
    cv2.waitKey(1)
