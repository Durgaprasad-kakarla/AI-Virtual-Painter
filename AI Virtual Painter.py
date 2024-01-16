import cv2
import mediapipe as mp
import numpy as np
import time
import mediapipe as mp
import HandTrackingModule as htm
import os
import numpy as np
folderPath="New Images"
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
drawColor=(0,140,255)
detector=htm.handDetector(detectionCon=0.9)
imgCanvas=np.zeros((570,1125,3),np.uint8)
# image=cv2.imread("New Images/1.png")
# image=cv2.resize(image,(1300,720))
cap=cv2.VideoCapture(0)
print(imgCanvas.shape)
while True:
    success,img=cap.read()
    img=cv2.flip(img,1)
    img = cv2.resize(img, (1300, 720))
    img2=img.copy()
    img2=cv2.resize(img2,(1125,570))
    header=cv2.resize(header,(1300,720))
    img[:720,:1300]=header
    img[150:720,175:1300]=img2
    detector.findHands(img2)
    lmList=detector.findPosition(img2)
    print(lmList)
    if len(lmList) != 0:
        # tip of index and middle fingers
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]
        fingers = detector.fingersUp()
        dist=detector.findDistance(8,12)
        print(dist)
        # If selection Mode--Two fingers are up
        if dist<170:
                xp, yp = 0, 0
                cv2.circle(img2, (x1, y1), 10, drawColor, cv2.FILLED)
                cv2.circle(img2, (x2, y2), 10, drawColor, cv2.FILLED)
                cv2.line(img2, (x1, y1), (x2, y2), drawColor, max(1, 10 // 3))
                print("Selection Mode")
                if y1 < 30:
                    if 100 < x1 < 200:
                        header = overlayList[0]
                        drawColor = (0,140,255)
                    elif 250 < x1 < 350:
                        header = overlayList[1]
                        drawColor = (0, 255, 255)
                    elif 450 < x1 < 600:
                        header = overlayList[2]
                        drawColor = (255, 0, 255)
                    elif 650 < x1 < 800:
                        header = overlayList[3]
                        drawColor = (0, 0, 255)
                    elif x1>800:
                        header=overlayList[7]
                        drawColor=(0,0,0)
                if x1<60:
                    if 100<y1<200:
                        header = overlayList[4]
                        drawColor = (255, 0, 0)
                    elif 300<y1<400:
                        header = overlayList[5]
                        drawColor = (0, 255,0)
                    elif y1>400:
                        header = overlayList[6]
                        drawColor = (153, 255, 153)
            # If drawing mode -- Index finger is up
        else:
            cv2.circle(img2, (x1, y1), 15, drawColor, cv2.FILLED)
            print("Drawing mode")
            if xp == 0 and yp == 0:
                xp, yp = x1, y1
            if drawColor == (0, 0, 0):
                cv2.line(img2, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
            else:
                cv2.line(img2, (xp, yp), (x1, y1), drawColor, brushThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)
            xp, yp = x1, y1
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img2 = cv2.bitwise_and(img2, imgInv)
    img2 = cv2.bitwise_or(img2, imgCanvas)
    header = cv2.resize(header,(1300,720))
    img[:720,:1300] = header
    img[150:720, 175:1300] = img2


    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    # cv2.putText(img, "FPS " + str(int(fps)), (20, 150), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    cv2.imshow("AI Virtual Painter", img)
    cv2.imshow("AI Virtual Painter canvas", imgCanvas)
    cv2.waitKey(1)

