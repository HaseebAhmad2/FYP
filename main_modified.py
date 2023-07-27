import cv2,sys
import numpy as np
import time
import mediapipe as mp
import math



class pDetector():
    def _init_(self, mode=False, upBody=False, smooth=True,detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.upBody, self.smooth,self.detectionCon, self.trackCon)

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
                                           self.mpPose.POSE_CONNECTIONS)
        return img

    def detect_lm_points(self, img, draw=True):
        self.point_list = []
        if self.results.pose_landmarks:
            for id, landmarks in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                # print(id, landmarks)
                cx, cy = int(landmarks.x * w), int(landmarks.y * h)
                self.point_list.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return self.point_list

    def findAngle(self, img, point1, point2, point3, draw=True):
        x1, y1 = self.point_list[point1][1:]
        x2, y2 = self.point_list[point2][1:]
        x3, y3 = self.point_list[point3][1:]
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
        if angle < 0:
            angle += 360
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
            cv2.line(img, (x3, y3), (x1, y1), (255, 255, 255), 3)
            cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x1, y1), 15, (0, 0, 255), 2)
            cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (0, 0, 255), 2)
            cv2.circle(img, (x3, y3), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 15, (0, 0, 255), 2)
            cv2.putText(img, str(int(angle)), (x2 - 50, y2 + 50),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        return angle

#name , picture path , 3 angle points , 2 angle ranges [2,3,4] are angle ranges and [5,6] are range of angles

def main(name,picture_path,video_path,angle1,angle2,angle3,range1,range2):
    cap = cv2.VideoCapture(video_path)
    detector = pDetector()
    count = 0
    dir = 0
    pTime = 0


    lst = []
    while True:
        success, img = cap.read()
        img = cv2.resize(img,(1920,1030))
        #img = cv2.flip(img,1)
        img2 = cv2.imread(f"poses/{picture_path}")
        img2 = cv2.resize(img2,(800,800))
        indentY,indentX = 0,350
        width,height,depth = img2.shape
        #img[indentY:indentY+height,indentX:indentX+width] = img2

        cv2.putText(img, "AI Yoga App", (890, 60), cv2.FONT_HERSHEY_SIMPLEX, 2,
                            (0, 0, 0), 4)
        text = ''
        for i in range(200):
            text+='_'
        cv2.putText(img, f"{text}", (0, 90), cv2.FONT_HERSHEY_SIMPLEX, 2,
                            (0, 0, 0), 4)
        cv2.rectangle(img, (0, 0), (400, 1000), (255, 255, 255), cv2.FILLED)
        cv2.putText(img, "Pose Like Above Image : ", (4, 430), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255), 4)
        cv2.putText(img, f"Pose Name : {name}", (4, 500), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (150, 100, 150), 2)


        cv2.putText(img, "Make sure your hands", (4, 540), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 110, 0), 2)

        cv2.putText(img, "and leg position is ", (4, 570), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 110, 0), 2)

        cv2.putText(img, "same as above", (4, 600), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 110, 0), 2)

        
        cv2.putText(img, "Wait until 100%", (4, 900), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 0, 0), 2)
        cv2.putText(img, "Perform next after", (4, 930), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 0, 0), 2)
        img2 = cv2.imread(f"poses/{picture_path}")
        img2 = cv2.resize(img2,(400,400))
        indentY,indentX = 0,0         
        width,height,depth = img2.shape
        img[indentY:indentY+height,indentX:indentX+width] = img2


        img = detector.findPose(img, True)
        point_list = detector.detect_lm_points(img, False)
        if len(point_list) != 0:
            threshold = 300
            angle = detector.findAngle(img, angle1,angle2,angle3)
            if angle>range1 and angle<range2:
                lst.append(angle)
                if len(lst)>threshold:
                    break
            else:
                cv2.putText(img, "Alert", (4, 720), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 0, 255), 3)
                cv2.putText(img, "Try to pose like", (4, 750), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (100, 255, 100), 2)
                cv2.putText(img, "shown in above image", (4, 780), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (100, 255, 100), 2)

            percentage = np.interp(len(lst), (0,threshold), (0, 100))
            bar = np.interp(len(lst), (0,threshold), (650, 100))
            color = (255, 0, 255)
            if percentage == 100:
                color = (0, 255, 0)
                if dir == 0:
                    count += 0.5
                    dir = 1
            if percentage == 0:
                color = (0, 255, 0)
                if dir == 1:
                    count += 0.5
                    dir = 0

            # Draw Bar
            cv2.rectangle(img, (1700, 100), (1775, 650), color, 2)
            cv2.rectangle(img, (1700, int(bar)), (1775, 650), color, cv2.FILLED)
            cv2.putText(img, f'{int(percentage)} %', (1700, 750), cv2.FONT_HERSHEY_SIMPLEX, 2,
                        color, 2)
            angle = round(angle)
            cv2.putText(img, f"Current Angle : {angle}", (4, 630), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (255, 110, 0), 2)

            cv2.putText(img, f"Angle to maintain:{range1}-{range2}", (4, 660), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (255, 110, 0), 2)

        cv2.imshow("Image", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            sys.exit(0)


# structure : all_data => name , picture path , 3 angle points , 2 angle ranges [2,3,4] are angle ranges and [5,6] are range of angles


all_data = {
    'Crescent Lunge Pose':['Crescent Lunge Pose','1_Crescent Lunge Pose.jpg','1.mp4',16,25,28,70,87],
    'Dancer Pose':['Dancer Pose','3_Dancer Pose.jpg','3.mp4',16,28,27,45,55],
    'Cobra Pose':['Cobra Pose','4_cobra pose.jpg','4.mp4',27,15,11,280,300],
    'Worrior 2':['Worrior 2','5_worrior2.jpg','5.mp4',24,16,26,25,35],
    'Chair':['Chair','6_chair.jpg','',23,25,27,40,90],
    'Boat Pose':['Boat Pose','7_Boat pose.jpg','',12,24,28,45,60],
    'Reversed Corpse Pose':['Reversed Corpse Pose','8_Reversed Corpse Pose.jpg','',0,24,28,175 ,190],
    'Half Forward Bend':['Half Forward Bend','9_Half Forward Bend.jpg','',12,24,28,70,90],
    'Warrior 3':['Warrior 3','10_Warrior 3.jpg','',27,23,28,70,90],
    'half moon pose':['half moon pose','11_half moon pose.jpg','',0,24,28,70,90],
    'half Boat pose':['half Boat pose','12_half Boat pose.jpg','',12,24,26,45,60],
    #'Half Lotus pose':['Half Lotus pose','13_Half Lotus pose.jpg','',12,24,26,45,60],
    'legs up the wall':['legs up the wall','14_legs up the wall.jpg','',1,24,28,70,90],
    'Mountain pose':['Mountain pose','15_Mountain pose.jpg','',1,24,28,175,190],
    'downward dog pose':['downward dog pose','16_downward dog pose.jpg','',11,23,27,65,80],
    'Half farword bend':['Half farword bend','17_Half farword bend.jpg','',28,24,12,80,98],
    'plank pose':['plank pose','18_plank pose.jpg','',12,24,28,175,190],
    'staff pose':['staff pose','19_staff pose.jpg','',11,23,27,75,90],
    'head to knee pose':['head to knee pose','21_head to knee pose.jpg','',12,24,28,55,70],
    'seated farword bend':['seated farword bend pose','22_seated farword bend.jpg','',12,24,28,65,80],
    'corpse pose':['corpse pose','23_corpse pose.jpg','',12,24,28,175,190],
    
    
}


path = int(input('''
    Press 1 - Run from camera that is live detection, 
    press 2 - Run on video, 
    press 3 - Run all poses with camera feed, 
    press 4 - Run all poses from video feed.
    '''))
if path==1:
    rep_dic = {
    1:'Crescent Lunge Pose',
    2:'Dancer Pose',
    3:'Cobra Pose',
    4:'Worrior 2',
    5:'Chair',
    6:'Boat Pose',
    7:'Reveresed Corpse Pose',
    8:'Half Farword Bend Pose',
    9:'Warrior 3',
    10:'half moon pose',
    11:'half Boat pose',
    #12:'Half Lotus pose',
    12:'legs up the wall',
    13:'Mountain',
    14:'downward dog pose',
    15:'Half farword bend',
    16:'plank pose',
    17:'staff pose',
    18:'head to knee pose',
    19:'seated farword bend',
    20:'corpse pose',
    }
    print(rep_dic)
    pose_ = int(input('Enter pose : '))
    pose_ = rep_dic[pose_]
    pose_ = all_data[pose_]
    #for i in all_data.values():
    main(pose_[0],pose_[1],0,pose_[3],pose_[4],pose_[5],pose_[6],pose_[7])

elif path==2:
    rep_dic = {
    1:'Crescent Lunge Pose',
    2:'Dancer Pose',
    3:'Cobra Pose',
    4:'Worrior 2',
    5:'Chair',
    6:'Boat Pose',
    7:'Reveresed Corpse Pose',
    8:'Half Farword Bend Pose',
    9:'Warrior 3',
    10:'half moon pose',
    11:'half Boat pose',
    #12:'Half Lotus pose',
    12:'legs up the wall',
    13:'Mountain',
    14:'downward dog pose',
    15:'Half farword bend',
    16:'plank pose',
    17:'staff pose',
    18:'head to knee pose',
    19:'seated farword bend',
    20:'corpse pose',
    }
    print(rep_dic)
    pose_ = int(input('Enter pose : '))
    pose_ = rep_dic[pose_]
    pose_ = all_data[pose_]

    main(pose_[0],pose_[1],pose_[2],pose_[3],pose_[4],pose_[5],pose_[6],pose_[7])
elif path==3:
    for i in all_data.values():
        main(i[0],i[1],0,i[3],i[4],i[5],i[6],i[7])

elif path==4:
    for i in all_data.values():
        main(i[0],i[1],i[2],i[3],i[4],i[5],i[6],i[7])


