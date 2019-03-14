import cv2
import sys

class FaceDetectionCv:

    def __init__(self,config_dir):
        self.cascPath = config_dir+"/haarcascade_frontalface_default.xml"
        # Create the haar cascade
        self.faceCascade = cv2.CascadeClassifier( self.cascPath)


    def processImg(self,img_gray):
        # Detect faces in the image
        self.faceCascade = cv2.CascadeClassifier(self.cascPath)
        faces=[]
        try:
            #faces = self.faceCascade.detectMultiScale(
            #    img_gray,
            #    scaleFactor=1.1,
            #    minNeighbors=5,
            #    minSize=(30, 30),
            #    flags = cv2.CASCADE_SCALE_IMAGE
            #)

            faces = self.faceCascade.detectMultiScale(
                img_gray,
                scaleFactor=1.3,
                minNeighbors=5
            )
            #flags = 
        except  cv2.error as e:
            print "----------------------------------------------------------------------------------------------------------------------------------------------->exception occurs during face detection process"   
        print("Found {0} faces!".format(len(faces)))

        result=[]
        
        for (x, y, w, h) in faces:
            result.append(( long(y) ,long(x+h) ,long(y+w) ,long(x) ))
            
        return result



