#! /usr/bin/env python
__author__ ='Jacques Saraydaryan'
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
from people_face_identification.srv import LearnFaceFromImg,DetectFaceFromImg
from process.FaceDetectionCv import FaceDetectionCv

def TestFaceDetectionCV():
    _bridge = CvBridge()
    topic_img=rospy.get_param('PeopleFaceIdentificationSimple/topic_img','/usb_cam/image_raw')
    test_folder=rospy.get_param('PeopleFaceIdentificationSimple/imgtest_folder','../data/img_tests')
    config_folder=rospy.get_param('PeopleFaceIdentificationSimple/config_folder','../config')

    
    pub = rospy.Publisher(topic_img, Image, queue_size=10)
    rospy.init_node('TestFaceDetectionCV', anonymous=True)


    
    #Load Image

    #img_loaded2 = cv2.imread(test_folder+'/onePeople.jpg')
    #img_loaded2 = cv2.imread(test_folder+'/imgMulti4.png')
    img_loaded2 = cv2.imread(test_folder+'/1BigSeveralPeople.png')
    #img_loaded2 = cv2.imread(test_folder+'/group-diff-position.jpg')
    #img_loaded2 = cv2.imread(test_folder+'/imageFrontPepper4.png')
    
    msg_im2 = cv2.cvtColor(img_loaded2, cv2.COLOR_BGR2GRAY)
    
    
    faceDetection = FaceDetectionCv(config_folder)

    result=faceDetection.processImg(msg_im2)

    for  (top, right, bottom, left) in result:
       cv2.rectangle(img_loaded2, ( left,top), ( right,bottom), (0, 255, 0), 2)

    cv2.imshow("Faces found", img_loaded2)
    cv2.waitKey(0)

    # spin
    rospy.spin()

if __name__ == '__main__':
        TestFaceDetectionCV()
