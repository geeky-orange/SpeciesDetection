'''
Created By Muhammad Mubeen
Using dataset from Pre-trained CoCo Library
Using CNN and darkNet model to predit and train the models 
'''
import cv2


classNames = []

# Change to relative path when migrating to resp pi

# classFile = "/home/pi/Desktop/Object_Detection_Files/coco.names"
classFile = "coco.names"

with open(classFile,"rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

configPath = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "frozen_inference_graph.pb"

# Config and setting up the hyper parameters
net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Set the drawn box color and high to adjust according to the object
def getObjects(img, thres, nms, draw=True, objects=[]):
    classIds, confs, bbox = net.detect(img,confThreshold=thres,nmsThreshold=nms)
    #print(classIds,bbox)
    if len(objects) == 0: objects = classNames
    objectInfo =[]
    if len(classIds) != 0:
        for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
            className = classNames[classId - 1]
            if className in objects:
                objectInfo.append([box,className])
                if (draw):
                    cv2.rectangle(img,box,color=(0,255,0),thickness=2)
                    cv2.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+30),
                    cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                    cv2.putText(img,str(round(confidence*100,2)),(box[0]+200,box[1]+30),
                    cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

    return img,objectInfo

# determines the size of the live feed window
if __name__ == "__main__":

# Add the recorded file
    cap = cv2.VideoCapture('Horse_riding_3.mp4')
    cap.set(3,640)
    cap.set(4,480)
    #cap.set(10,70)

# Detects the object in the video recorded file
    while True:
        success, img = cap.read()
        result, objectInfo = getObjects(img,0.45,0.2)
        #print(objectInfo)
        cv2.imshow("Output",img)
        cv2.waitKey(1)
