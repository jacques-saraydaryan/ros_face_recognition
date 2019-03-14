#! /usr/bin/env python
__author__ ='Jacques Saraydaryan'
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
from people_face_identification.srv import LearnFaceFromImg,DetectFaceFromImg

def LoadImgAndPublish():
    _bridge = CvBridge()
    topic_img=rospy.get_param('PeopleFaceIdentificationSimple/topic_img','/usb_cam/image_raw')
    test_folder=rospy.get_param('PeopleFaceIdentificationSimple/imgtest_folder','../data/img_tests')

    
    pub = rospy.Publisher(topic_img, Image, queue_size=10)
    rospy.init_node('LoadAndPublishImg', anonymous=True)


    
    #Load Image
    #img_loaded1 = cv2.imread(test_folder+'/1BigSeveralPeople.png')
    img_loaded1 = cv2.imread(test_folder+'/1BigSeveralPeople.png')
    
    msg_im1 = _bridge.cv2_to_imgmsg(img_loaded1, encoding="bgr8")

    #img_loaded2 = cv2.imread(test_folder+'/onePeople.jpg')
    #img_loaded2 = cv2.imread(test_folder+'/onePeople.jpg')
    #img_loaded2 = cv2.imread(test_folder+'/imgMulti4.png')
    #img_loaded2 = cv2.imread(test_folder+'/group-diff-position.jpg')
    img_loaded2 = cv2.imread(test_folder+'/group-sit.jpg')
    #img_loaded2 = cv2.imread(test_folder+'/imageFrontPepper4.png')
    
    msg_im2 = _bridge.cv2_to_imgmsg(img_loaded2, encoding="bgr8")
    
    
    #call service to learn people
    rospy.wait_for_service('learn_face_from_img')
    try:
        learn_from_img_srv = rospy.ServiceProxy('learn_face_from_img', LearnFaceFromImg)
        resp1 = learn_from_img_srv("HOwaRD!!!",msg_im1)
        print "service:"+str(resp1.result)
    except rospy.ServiceException, e:
        print "Service call failed: %s"%e

    #Publish image
    pub.publish(msg_im1)

    rospy.sleep(5)

    #call service to detect people
    rospy.wait_for_service('detect_face_from_img')
    try:
        detect_from_img_srv = rospy.ServiceProxy('detect_face_from_img', DetectFaceFromImg)
        resp2 = detect_from_img_srv(msg_im2,False)
        print "service:"+str(resp2.entityList)
    except rospy.ServiceException, e:
        print "Service call failed: %s"%e

    
    pub.publish(msg_im2)
    



    # spin
    rospy.spin()

if __name__ == '__main__':
    try:
        LoadImgAndPublish()
    except rospy.ROSInterruptException:
        pass