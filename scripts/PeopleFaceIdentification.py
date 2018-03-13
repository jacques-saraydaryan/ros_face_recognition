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
from robocup_msgs.msg import Entity2D


import cv2
from Queue import Queue,Full
from threading import Thread, Lock

import time
import uuid


import face_recognition
from common import Face,Timeout

#TO DO BEFORE LAUNCH
# export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64:$LD_LIBRARY_PATH
#


class PeopleFaceIdentification():
    nb_thread=5
    queue_in_size=5
    queue_out_size=1
    continuous_learn=True
    learn_timeout=20
    # define the current image process, DECTECTION or LEARNING
    STATUS='DETECTION'
    FACE_FOLDER=        '/home/jsaraydaryan/ros_robotcupathome_ws/src/people_management/people_face_identification/data/labeled_people'
    FACE_FOLDER_AUTO=   '/home/jsaraydaryan/ros_robotcupathome_ws/src/people_management/people_face_identification/data/auto_labeled_people'
    #FACE_FOLDER='../data/labeled_people'
    user_cnn_module=False
    topic_img='/usb_cam/image_raw'
    topic_face_img='/face_detection/face_image'
    topic_face_box='/face_detection/face_msg'
    publish_img=True
    activate_detection=True
    faceList={}

    #@profile
    def __init__(self):
        rospy.init_node('people_face_identification', anonymous=False)
        self._bridge = CvBridge()
        rospy.loginfo('CV bridge')
        self.configure()

        # Subscribe to the face positions
        # Subscribe to the face positions
        self.sub_rgb = rospy.Subscriber(self.topic_img, Image, self.rgb_callback, queue_size=20)
        self.pub_detections_image = rospy.Publisher(self.topic_face_img, Image, queue_size=20)
        self.pub_detections_msg = rospy.Publisher(self.topic_face_box, Entity2D, queue_size=1)
        self.learnFaceSrv = rospy.Service('learn_face', LearnFace, self.learnFaceSrvCallback)
        # spin
        rospy.spin()
    #@profile
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

        rospy.loginfo("Param: face_folder_auto:"+str(self.FACE_FOLDER_AUTO))
        rospy.loginfo("Param: face_folder:"+str(self.FACE_FOLDER))
        rospy.loginfo("Param: user_cnn_module:"+str(self.user_cnn_module))
        rospy.loginfo("Param: continuous_learn:"+str(self.continuous_learn))
        rospy.loginfo("Param: topic_img:"+str(self.topic_img))
        rospy.loginfo("Param: topic_face_img:"+str(self.topic_face_img))
        rospy.loginfo("Param: topic_face_box:"+str(self.topic_face_box))
        rospy.loginfo("Param: publish_img:"+str(self.publish_img))
        rospy.loginfo("Param: activate_detection:"+str(self.activate_detection))


        #load face files form data directory
        self.loadLearntFaces()

        self.input_q = Queue(self.queue_in_size)  # fps is better if queue is higher but then more lags
        self.output_q = Queue(self.queue_out_size)

        #t_ou = Thread(target=self.worker_out, args=())
        #t_ou.daemon = True
        #t_ou.start()
        #rospy.loginfo("worker OUT started..")
        self.mutex = Lock()
        for i in range(self.nb_thread):
            t = Thread(target=self.worker, args=(self.input_q, self.output_q,i))
            t.daemon = True
            t.start()
            rospy.loginfo("worker %s started..",str(i))


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
                    face_encoding = face_recognition.face_encodings(face_image)[0]
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
    #@profile
    def rgb_callback(self, data):
        
        #"""
        #Callback for RGB images
        #"""
        
        if self.activate_detection:
            try:
                self.input_q.put_nowait(data)
            except Full as e:
                rospy.logwarn(e)
        
    #@profile
    def worker_out(self):
            while True:
                if self.output_q.empty():
                    time.sleep(0.05)
                    rospy.logdebug("----------WORKER[OUT] queue is empty")
                    pass  # fill up queue
                else:
                    frame = self.output_q.get()
                    msg_im = self._bridge.cv2_to_imgmsg(frame, encoding="bgr8")
                    self.pub_detections_image.publish(msg_im)
    #@profile
    def worker(self,input_q, output_q,name_w):
        while True:
            try:
                data = self.input_q.get(0.05)
                if data != None:

                    data_result, label, top,left,bottom,right=self.process_img(data,input_q, output_q,name_w)
                    if( label != 'NONE'):
                        detected_face=Entity2D()
                        detected_face.label=label
                        x0,y0=self.processBoxCenter(left,top,right,bottom)
                        detected_face.pose.x=x0
                        detected_face.pose.y=y0
                        self.pub_detections_msg.publish(detected_face)


                    if(self.publish_img):
                        msg_im = self._bridge.cv2_to_imgmsg(data_result, encoding="bgr8")
                        self.pub_detections_image.publish(msg_im)
                    #output_q.put(data_result)
            except Full as e:
                rospy.logwarn(e)
                
            
    #@profile      
    def process_img(self,data,input_q, output_q,name_w):
            new_learnt_face=[]
            label_r='NONE'
            try:
                # Conver image to numpy array
                frame = self._bridge.imgmsg_to_cv2(data, 'bgr8')
                if(self.user_cnn_module):
                    face_locations = face_recognition.face_locations(frame, number_of_times_to_upsample=0, model="cnn")
                else:
                    face_locations = face_recognition.face_locations(frame)
                i=0
                for location in face_locations:
                    #rospy.loginfo("WORKER[%s] ----face[%s] x0: %s, y0: %s, x1: %s, y1: %s",str(name_w),str(i),str(location[0]),str(location[1]),str(location[2]),str(location[3]))
                    i=i+1

                face_encodings = face_recognition.face_encodings(frame, face_locations)
                # Find all the faces and face enqcodings in the frame of video
                top_r=0
                bottom_r=0
                left_r=0
                right_r=0

                # Loop through each face in this frame of video
                for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                    # See if the face is a match for the known face(s)

                    name = self._processDetectFace(top, right, bottom, left,face_encoding)
                    rospy.loginfo("STATUS: "+self.STATUS)
                    if (self.STATUS=='LEARNING' and name == "Unknown") or (self.continuous_learn and name == "Unknown"):
                        label_tmp=str(uuid.uuid1())
                        rospy.loginfo("unkwon face: launch learn operation")
                        self._processLearnFace(top, right, bottom, left,face_encoding,label_tmp,frame,new_learnt_face)

                    label_r=name
                    # Draw a box around the face
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                    # Draw a label with a name below the face
                    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

                    x0,y0 =self.processBoxCenter(top, right, bottom, left)
                    #print('x0:'+str(x0)+', y0:'+str(y0)+',top:'+str(top)+',left:'+str(left)+',bottom:'+str(bottom)+',right:'+str(right))
                    cv2.circle(frame, (x0, y0), 5, (0, 255, 0), cv2.FILLED)
                    top_r=top
                    bottom_r=bottom
                    left_r=left
                    right_r=right

                    #return frame,name,top,left,bottom,right

                #check the biggest face learnt
                if not self.continuous_learn:
                    max_box=0
                    biggest_face=None
                    for f in new_learnt_face:
                        if max_box<f.size:
                            max_box=f.size
                            biggest_face=f
                    if len(new_learnt_face)>0:
                        self.mutex.acquire()
                        try:
                            self.STATUS!='LEARNT'
                            oldId=biggest_face.label
                            del self.faceList[oldId]
                            biggest_face.label=self.labelToLearn
                            self.faceList[self.labelToLearn]=biggest_face
                            rospy.loginfo("")
                            os.rename(self.FACE_FOLDER_AUTO+"/"+oldId+'.png', self.FACE_FOLDER_AUTO+"/"+self.labelToLearn+'.png')
                            rospy.loginfo("BIGGEST FACE of "+str(len(new_learnt_face))+":"+biggest_face.label)
                        finally:
                            self.mutex.release()

                return frame,label_r,top_r,left_r,bottom_r,right_r

                
                   
            except CvBridgeError as e:
                    rospy.logwarn(e)
                    return "no Value"
                        #time.sleep(10)

    def _processLearnFace(self,top, right, bottom, left,face_encoding,label_tmp,frame,new_learnt_face):
        #self.mutex.acquire()
        if self.mutex.acquire(False):
            try:
                #save file to learn directory and crop according box
                cv2.imwrite(self.FACE_FOLDER_AUTO+"/"+label_tmp+".png", frame[top:bottom, left:right])
                new_face=Face.Face(0,0,0,0,label_tmp)
                new_face.encode(face_encoding)
                self.faceList[label_tmp]=new_face
                new_learnt_face.append(new_face)
            finally:
                self.mutex.release()
        else:
            rospy.loginfo(" lock is already aquired, learning is already in progress")
        

    def _processDetectFace(self,top, right, bottom, left,face_encoding):
        name = "Unknown"
        for label in self.faceList.keys():
            match = face_recognition.compare_faces([self.faceList[label].encoding], face_encoding)
            if match[0]:
                name = self.faceList[label].label
        return name

    def processBoxCenter(self, top, right, bottom, left):
        y0= int(top+abs((bottom-top)/2))
        x0= int(left+abs((right-left)/2))
        return x0,y0
    
    def learnFaceSrvCallback(self,req):
        self.labelToLearn=req.label
        rospy.loginfo("Changing status from "+self.STATUS+" to LEARNING ")
        self.STATUS='LEARNING'
        error=True
        start = time.time()
        try:
                
                while (time.time() - start < self.learn_timeout) and self.STATUS!='LEARNT':
                    time.sleep(0.05)
                
                if(self.STATUS=='LEARNT'):        #data_result=self.process_img(data,self.input_q, self.output_q,"test")       
        #msg_im = self._bridge.cv2_to_imgmsg(data_result, encoding="bgr8")
        #self.pub_detections_image.publish(msg_im)
                    error=False
        finally:            
            self.STATUS='DETECTION'
            if error:
                rospy.logwarn("end learn service with error (may be due to a time out ?)")
                return False
            else:
                rospy.loginfo("end learn service with SUCCESS")
                return True

    

def main():
    #""" main function
    #"""
    node = PeopleFaceIdentification()

if __name__ == '__main__':
    main()