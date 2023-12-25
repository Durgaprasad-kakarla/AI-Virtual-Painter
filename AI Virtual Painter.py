import cv2
import numpy as np
import time
import mediapipe as mp
import HandtrackingModule as htm
import os
import numpy as np
folderPath="Paints"
myList=os.listdir(folderPath)
print(myList)
overlayList=[]

brushThickness=8
eraserThickness=50
xp,yp=0,0

for imPath in myList:
    image=cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)
header=overlayList[0]
print(len(overlayList))
cap=cv2.VideoCapture(0)
pTime=0
drawColor=(255,0,255)
detector=htm.handDetector(detectionCon=0.9)
imgCanvas=np.zeros((720,1100,3),np.uint8)
while True:
    success,img=cap.read()
    img=cv2.resize(img,(1100,720))
    img=cv2.flip(img,1)
    header = cv2.resize(header, (1100, 115))
    detector.findHands(img)
    lmList=detector.findPosition(img,draw=False)
    if len(lmList)!=0:
        #tip of index and middle fingers
        x1,y1=lmList[8][1:]
        x2,y2=lmList[12][1:]
        fingers=detector.fingersUp()

        #If selection Mode--Two fingers are up
        if fingers[1] and fingers[2]:
            xp,yp=0,0
            cv2.rectangle(img,(x1,y1-25),(x2,y2+25),(255,0,255),cv2.FILLED)
            print("Selection Mode")
            if y1<116:
                if 0<x1<320:
                    header=overlayList[0]
                    drawColor=(255,0,255)
                elif 350<x1<550:
                    header=overlayList[1]
                    drawColor = (255, 0, 0)
                elif 600<x1<850:
                    header=overlayList[2]
                    drawColor = (0, 0, 255)
                else:
                    header=overlayList[3]
                    drawColor=(0,0,0)
            cv2.rectangle(img,(x1,y1-25),(x2,y2+25),drawColor,cv2.FILLED)

        #If drawing mode -- Index finger is up
        if fingers[1] and fingers[2]==False:
            cv2.circle(img,(x1,y1),15,drawColor,cv2.FILLED)
            print("Drawing mode")
            if xp==0 and yp==0:
                xp,yp=x1,y1
            if drawColor==(0,0,0):
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
            else:
                cv2.line(img,(xp,yp),(x1,y1),drawColor,brushThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)
            xp,yp=x1,y1

    imgGray=cv2.cvtColor(imgCanvas,cv2.COLOR_BGR2GRAY)
    _,imgInv=cv2.threshold(imgGray,50,255,cv2.THRESH_BINARY_INV)
    imgInv=cv2.cvtColor(imgInv,cv2.COLOR_GRAY2BGR)
    img=cv2.bitwise_and(img,imgInv)
    img=cv2.bitwise_or(img,imgCanvas)
    header=cv2.resize(header,(1100,115))
    img[0:115, 0:1100] = header
    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    cv2.putText(img,"FPS "+str(int(fps)),(20,150),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
    cv2.imshow("AI Virtual Painter",img)
    cv2.imshow("AI Virtual Painter canvas", imgCanvas)
    cv2.waitKey(1)
