#! /usr/bin/env python
__author__ ='Jacques Saraydaryan'
import os
import sys

import numpy as np
from cv_bridge import CvBridge, CvBridgeError

#sys.path.append(os.path.dirname(__file__) + "/../pose-tensorflow/")
#sys.path.append("/opt/ros/kinetic/lib/python2.7/dist-packages/")
# ROS
import rospy
from sensor_msgs.msg import Image
from people_face_identification.srv import *
from robocup_msgs.msg import Entity2D,Entity2DList,Box


import cv2
import time
import uuid

import face_recognition

from common import Face,Timeout

#TO DO BEFORE LAUNCH
# export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64:$LD_LIBRARY_PATH
#



class PeopleFaceIdentificationSimple():
    continuous_learn=True
    learn_timeout=20
    # define the current image process, DECTECTION or LEARNING
    LABEL_FACE="FACE"
    LEARNT_STATUS='LEARNT'
    LEARNING_STATUS='LEARNING'
    DETECTION_STATUS='DETECTION'
    FORCE_LEARNING_STATUS='FORCE_LEARNING'
    STATUS='DETECTION'
    FACE_FOLDER=        '/home/jsaraydaryan/ros_robotcupathome_ws/src/people_management/people_face_identification/data/labeled_people'
    FACE_FOLDER_AUTO=   '/home/jsaraydaryan/ros_robotcupathome_ws/src/people_management/people_face_identification/data/auto_labeled_people'
    user_cnn_module=False
    topic_img='/usb_cam/image_raw'
    topic_face_img='/face_detection/face_image'
    topic_face_box='/face_detection/face_msg'
    topic_all_faces_img='/face_detection/all_faces_image'
    topic_all_faces_box='/face_detection/all_faces_msg'
    publish_img=True
    activate_detection=True
    only_detect_faces=False
    labelToLearn="default_label"
   
    #FACE_FOLDER='../data/labeled_people'
    faceList={}
   
    def __init__(self):
        rospy.init_node('people_face_identification_simple', anonymous=False)
        self._bridge = CvBridge()
        rospy.loginfo('CV bridge')
        self.configure()

        # Subscribe to the face positions
        self.sub_rgb = rospy.Subscriber(self.topic_img, Image, self.rgb_callback, queue_size=20)
        self.pub_detections_image = rospy.Publisher(self.topic_face_img, Image, queue_size=20)
        self.pub_detections_msg = rospy.Publisher(self.topic_face_box, Entity2DList, queue_size=1)

        self.pub_all_faces_detections_image = rospy.Publisher(self.topic_all_faces_img, Image, queue_size=20)
        self.pub_all_faces_detections_msg = rospy.Publisher(self.topic_all_faces_box, Entity2DList, queue_size=1)

        
        self.learnFaceSrv = rospy.Service('learn_face', LearnFace, self.learnFaceSrvCallback)
        self.learnFaceFromImgSrv = rospy.Service('learn_face_from_img', LearnFaceFromImg, self.learnFaceFromImgSrvCallback)
        self.detectFaceFromImgSrv = rospy.Service('detect_face_from_img', DetectFaceFromImg, self.detectFaceFromImgSrvCallback)
        self.toogleFaceDetectionSrv = rospy.Service('toogle_face_detection', ToogleFaceDetection, self.toogleFaceDetectionSrvCallback)
        self.toogleAutoLearnFaceSrv = rospy.Service('toogle_auto_learn_face', ToogleAutoLearnFace, self.toogleAutoLearnFaceSrvCallback)
        self.getImgFomIdSrv = rospy.Service('get_img_from_id', GetImgFromId, self.getImgFromIdSrvCallback)

        # spin
        rospy.spin()
    
    
    #######################################################################
    #######                Configure Current Node                    ######
    #######################################################################
    def configure(self):
        #load face files form data directory
        self.FACE_FOLDER=rospy.get_param('PeopleFaceIdentificationSimple/face_folder')
        self.FACE_FOLDER_AUTO=rospy.get_param('PeopleFaceIdentificationSimple/face_folder_auto')
        self.user_cnn_module=rospy.get_param('PeopleFaceIdentificationSimple/user_cnn_module')
        self.continuous_learn=rospy.get_param('PeopleFaceIdentificationSimple/continuous_learn')
        self.learn_timeout=rospy.get_param('PeopleFaceIdentificationSimple/learn_timeout')

        self.topic_img=rospy.get_param('PeopleFaceIdentificationSimple/topic_img')
        self.topic_face_img=rospy.get_param('PeopleFaceIdentificationSimple/topic_face_img')
        self.topic_face_box=rospy.get_param('PeopleFaceIdentificationSimple/topic_face_box')
        self.publish_img=rospy.get_param('PeopleFaceIdentificationSimple/publish_img')
        self.activate_detection=rospy.get_param('PeopleFaceIdentificationSimple/activate_detection')

        self.topic_all_faces_img=rospy.get_param('PeopleFaceIdentificationSimple/topic_all_faces_img')
        self.topic_all_faces_box=rospy.get_param('PeopleFaceIdentificationSimple/topic_all_faces_box')

        self.only_detect_faces=rospy.get_param('PeopleFaceIdentificationSimple/only_detect_faces')
        

        rospy.loginfo("Param: face_folder_auto:"+str(self.FACE_FOLDER_AUTO))
        rospy.loginfo("Param: face_folder:"+str(self.FACE_FOLDER))
        rospy.loginfo("Param: user_cnn_module:"+str(self.user_cnn_module))
        rospy.loginfo("Param: continuous_learn:"+str(self.continuous_learn))
        rospy.loginfo("Param: topic_img:"+str(self.topic_img))
        rospy.loginfo("Param: topic_face_img:"+str(self.topic_face_img))
        rospy.loginfo("Param: topic_face_box:"+str(self.topic_face_box))
        rospy.loginfo("Param: activate_detection:"+str(self.activate_detection))
        rospy.loginfo("Param: only_detect_faces:"+str(self.only_detect_faces))
        self.publish_img=rospy.get_param('PeopleFaceIdentificationSimple/publish_img')

        self.loadLearntFaces()
        rospy.loginfo('configure ok')
    
    def loadLearntFaces(self):   
        path=self.FACE_FOLDER
        if os.path.exists(path):
            fileList=os.listdir(path)
            for file in fileList:
                label=os.path.splitext(file)[0]
                if label in  self.faceList:
                    rospy.logwarn("DUPICATE FACE LABEL file_name:"+file+", label:"+str(label))
                if(file!='.gitkeep'):
                    rospy.loginfo("file_name:"+file+", label:"+str(label))
                    #print("check --------------->:"+str(path+file))
                    face_image = face_recognition.load_image_file(path+"/"+file)
                    face_recognition_list=face_recognition.face_encodings(face_image)
                    rospy.loginfo(len(face_recognition_list))
                    if len(face_recognition_list)>0:
                        face_encoding = face_recognition_list[0]
                        current_face=Face.Face(0,0,0,0,label)
                        current_face.encode(face_encoding)
                        self.faceList[label]=current_face
        else:
             rospy.logerr("Unable to load face references, no such directory: "+str(path))
             return


    def shutdown(self):
        #"""
        #Shuts down the node
        #"""
        rospy.signal_shutdown("See ya!")



    #######################################################################
    #######         Callback for RGB imagesisFaceDetection           ######
    #######################################################################
    def rgb_callback(self, data):
        if self.activate_detection:
            if self.only_detect_faces:
                data_result, detected_faces_map= self.process_img_faces_only(data)
                self._publishOnlyFaceRosMsg(data,data_result,detected_faces_map)                
            else:
                data_result, label, top,left,bottom,right,detected_faces_list=self.process_img(data,None, None)
                self._publishRosMsg(data,data_result,detected_faces_list)
    
    #######################################################################
    #######                  Process only Faces                      ######
    #######################################################################
    def process_img_faces_only(self,data):
        detected_faces_map={}
        try:
                # Conver image to numpy array
                frame = self._bridge.imgmsg_to_cv2(data, 'bgr8')
                frame_copy = self._bridge.imgmsg_to_cv2(data, 'bgr8')
                if(self.user_cnn_module):
                    face_locations = face_recognition.face_locations(frame, number_of_times_to_upsample=0, model="cnn")
                else:
                    face_locations = face_recognition.face_locations(frame)
                i=0
                for location in face_locations:
                    detected_faces_map[i]=(location[0], location[1], location[2], location[3])
                    i=i+1
                    # Draw a box around the face
                    cv2.rectangle(frame, (location[3], location[0]), (location[1], location[2]), (0, 255, 0), 2)              
                return frame,detected_faces_map
        except CvBridgeError as e:
                    rospy.logwarn(e)
                    return "no Value"
                        #time.sleep(10)

    #######################################################################
    #######                 Process Images                           ######
    #######################################################################
    def process_img(self,data,name_w,current_status):
        return self.process_img_face(data,name_w,current_status,False)

    def process_img_face(self,data,name_w,current_status,isImgFace):
            detected_faces_list=[]
            new_learnt_face=[]
            face_locations=[]
            label_r='NONE'
            if(current_status== None):
                current_status=self.STATUS
            try:
                # Conver image to numpy array
                frame = self._bridge.imgmsg_to_cv2(data, 'bgr8')
                frame_copy = self._bridge.imgmsg_to_cv2(data, 'bgr8')
                
                if(self.user_cnn_module):
                    face_locations = face_recognition.face_locations(frame, number_of_times_to_upsample=0, model="cnn")
                else:
                    face_locations = face_recognition.face_locations(frame)
                i=0
                for location in face_locations:
                    i=i+1
                    
                if isImgFace:
                    face_encodings = face_recognition.face_encodings(frame)
                else:
                    face_encodings = face_recognition.face_encodings(frame, face_locations)

                # Find all the faces and face enqcodings in the frame of video
                top_r=0
                bottom_r=0
                left_r=0
                right_r=0             

                # Loop through each face in this frame oisFaceDetectionf video
                for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                    # See if the face is a match for the known face(s)
                    
                    name = self._processDetectFace(top, right, bottom, left,face_encoding)
                    current_face=Face.Face(top,left,bottom,right,name)
                    detected_faces_list.append(current_face)
                    
                    rospy.loginfo("STATUS: "+current_status)
                    if (current_status==self.LEARNING_STATUS and name == "Unknown") or (self.continuous_learn and name == "Unknown") or (current_status==self.FORCE_LEARNING_STATUS):
                        
                        label_tmp=str(uuid.uuid1())
                        rospy.loginfo("unkwon face: launch learn operation")
                        self._processLearnFace(top, right, bottom, left,face_encoding,label_tmp,frame_copy,new_learnt_face)
                    label_r=name
                    # Draw a box around the face
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                    # Draw a label with a name below the face
                    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

                    x0,y0 =self._processBoxCenter(top, right, bottom, left)
                    cv2.circle(frame, (x0, y0), 5, (0, 255, 0), cv2.FILLED)
                    top_r=top
                    bottom_r=bottom
                    left_r=left
                    right_r=right
                    #return frame,name,top,left,bottom,right
                
                #update label
                if name_w== None:
                    _labelToLearn=self.labelToLearn
                else:
                    _labelToLearn=name_w

                #check the biggest face learnt
                if not self.continuous_learn: 
                    max_box=0
                    biggest_face=None
                    for f in new_learnt_face:
                        if max_box<f.size:
                            max_box=f.size
                            biggest_face=f
                    if len(new_learnt_face)>0:
                        self.STATUS!='LEARNT'
                        oldId=biggest_face.label
                        del self.faceList[oldId]
                        biggest_face.label=_labelToLearn
                        self.faceList[_labelToLearn]=biggest_face
                        rospy.loginfo("")
                        os.rename(self.FACE_FOLDER_AUTO+"/"+oldId+'.png', self.FACE_FOLDER_AUTO+"/"+_labelToLearn+'.png')
                        rospy.loginfo("BIGGEST FACE of "+str(len(new_learnt_face))+":"+biggest_face.label)



                return frame,label_r,top_r,left_r,bottom_r,right_r,detected_faces_list      
            except CvBridgeError as e:
                    rospy.logwarn(e)
                    return "no Value"
                        #time.sleep(10)


    #######################################################################
    #######                 Publish Ros Msg                          ######
    #######################################################################
    def _publishRosMsg(self,data,data_result,detected_faces_list):
        eList=Entity2DList()
        entity2D_list=[]
        for  face in detected_faces_list:
            #rospy.loginfo("top: %s, right: %s, bottom: %s, left:%s",str(top), str(right), str(bottom), str(left))
                
            #publish boxes information
            detected_face=Entity2D()
            detected_face.header.frame_id=data.header.frame_id            
            detected_face.label=face.label
            x0,y0=self._processBoxCenter(face.left,face.top,face.right,face.bottom)
            detected_face.pose.x=x0
            detected_face.pose.y=y0

            box=Box()
            box.x=face.top
            box.y=face.left
            box.width=abs(face.left-face.right)
            box.height=abs(face.top-face.bottom)
            detected_face.bounding_box=box

            entity2D_list.append(detected_face)

        eList.entity2DList=entity2D_list
        self.pub_detections_msg.publish(eList)
                
        if(self.publish_img):
            msg_im = self._bridge.cv2_to_imgmsg(data_result, encoding="bgr8")
            msg_im.header.frame_id=data.header.frame_id
            self.pub_detections_image.publish(msg_im)

    #######################################################################
    #######            Publish Only Face Ros Msg                     ######
    #######################################################################
    def _publishOnlyFaceRosMsg(self,data,data_result,detected_faces_map):
        face_list=[]
        eList=Entity2DList()
        for  (top, right, bottom, left) in detected_faces_map.values():
            #rospy.loginfo("top: %s, right: %s, bottom: %s, left:%s",str(top), str(right), str(bottom), str(left))
                
            #publish boxes information
            detected_face=Entity2D()
            detected_face.header.frame_id=data.header.frame_id
           
            detected_face.label=self.LABEL_FACE
            x0,y0=self._processBoxCenter(left,top,right,bottom)
            detected_face.pose.x=x0
            detected_face.pose.y=y0

            box=Box()
            box.x=top
            box.y=left
            box.width=abs(left-right)
            box.height=abs(top-bottom)
            detected_face.bounding_box=box

            face_list.append(detected_face)
            #rospy.loginfo("msg sent:"+str(detected_face))
                    
        eList.entity2DList=face_list
        self.pub_all_faces_detections_msg.publish(eList)

        #publish image with detected faces if needed
        if(self.publish_img):
            msg_im = self._bridge.cv2_to_imgmsg(data_result, encoding="bgr8")
            msg_im.header.frame_id=data.header.frame_id
            self.pub_all_faces_detections_image.publish(msg_im)


    def _processLearnFace(self,top, right, bottom, left,face_encoding,label_tmp,frame,new_learnt_face):
        #save file to learn directory and crop according box
        cv2.imwrite(self.FACE_FOLDER_AUTO+"/"+label_tmp+".png", frame[top:bottom, left:right])
        new_face=Face.Face(0,0,0,0,label_tmp)
        new_face.encode(face_encoding)
        self.faceList[label_tmp]=new_face
        new_learnt_face.append(new_face)

        

    def _processDetectFace(self,top, right, bottom, left,face_encoding):
        name = "Unknown"
        for label in self.faceList.keys():
            match = face_recognition.compare_faces([self.faceList[label].encoding], face_encoding)
            if match[0]:
                name = self.faceList[label].label
        return name

    def _processBoxCenter(self, top, right, bottom, left):
        y0= int(top+abs((bottom-top)/2))
        x0= int(left+abs((right-left)/2))
        return x0,y0
    
    def learnFaceSrvCallback(self,req):
        self.labelToLearn=req.label
        rospy.loginfo("Changing status from "+self.STATUS+" to LEARNING ")
        self.STATUS=self.LEARNING_STATUS
        error=True
        start = time.time()
        try:
                
                while (time.time() - start < self.learn_timeout) and self.STATUS!=self.LEARNT_STATUS:
                    time.sleep(0.05)
                
                if(self.STATUS==self.LEARNT_STATUS):
                    error=False
        finally:            
            self.STATUS=self.DETECTION_STATUS
            if error:
                rospy.logwarn("end learn service with error (may be due to a time out ?)")
                return False
            else:
                rospy.loginfo("end learn service with SUCCESS")
                return True

    def detectFaceFromImgSrvCallback(self,req):
        img=req.img
        try:
            frame,label_r,top_r,left_r,bottom_r,right_r,detected_faces_list=self.process_img_face(img, None,self.DETECTION_STATUS,req.isImgFace)
            eList=Entity2DList()
            entity2D_list=[]
            for  face in detected_faces_list:
                #publish boxes information
                detected_face=Entity2D()
                detected_face.header.frame_id=img.header.frame_id            
                detected_face.label=face.label
                x0,y0=self._processBoxCenter(face.left,face.top,face.right,face.bottom)
                detected_face.pose.x=x0
                detected_face.pose.y=y0
                box=Box()
                box.x=face.top
                box.y=face.left
                box.width=abs(face.left-face.right)
                box.height=abs(face.top-face.bottom)
                detected_face.bounding_box=box
                entity2D_list.append(detected_face)
            eList.entity2DList=entity2D_list
            #-rospy.loginfo(str(eList))
            return DetectFaceFromImgResponse(eList)
        except:    
            rospy.loginfo("Service Detect face from Img end with error "+ str(sys.exc_info()[0]))
            return False

    def learnFaceFromImgSrvCallback(self,req):
        label=req.label
        img=req.img
        try:
            self.process_img(img, label,self.FORCE_LEARNING_STATUS)
            return True
        except:    
            rospy.loginfo("Service learn face from Img end with error "+ str(sys.exc_info()[0]))
            return False

    
    def toogleFaceDetectionSrvCallback(self,req):
        self.activate_detection=req.isFaceDetection
        return True

    def toogleAutoLearnFaceSrvCallback(self,req):
        self.continuous_learn=req.isAutoLearn
        return True

    def getImgFromIdSrvCallback(self,req):
        path=self.FACE_FOLDER
        auto_learn_path=self.FACE_FOLDER_AUTO
        imgResult=self.getImgFromLabel(path,req.label)
        if imgResult ==None:
            imgResult=self.getImgFromLabel(auto_learn_path,req.label)
        return GetImgFromIdResponse(imgResult)

    def getImgFromLabel(self,folder,label):
        try:
            if os.path.exists(folder):
                fileList=os.listdir(folder)
                for file in fileList:
                    label_f=os.path.splitext(file)[0]
                    if label_f ==  label:
                        #Load Image
                        img_loaded = cv2.imread((folder+"/"+file))
                        msg_img = self._bridge.cv2_to_imgmsg(img_loaded, encoding="bgr8")
                        return msg_img
        except Exception as e:
            rospy.logwarn("Error when searching image from label :"+str(e))
        return None


def main():
    #""" main function
    #"""
    node = PeopleFaceIdentificationSimple()

if __name__ == '__main__':
    main()