import mediapipe as mp
import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import csv

model_path="C:\\Academic\\courses\\opencv\\hand_landmarker.task"
base_options=python.BaseOptions(model_asset_path=model_path)
options=vision.HandLandmarkerOptions(base_options=base_options,num_hands=1)
detector=vision.HandLandmarker.create_from_options(options)
capture=cv2.VideoCapture(0,cv2.CAP_DSHOW)

connections=[
(0,1),(1,2),(2,3),(3,4),
(0,5),(5,6),(6,7),(7,8),
(5,9),(9,10),(10,11),(11,12),
(9,13),(13,14),(14,15),(15,16),
(13,17),(17,18),(18,19),(19,20),
(0,17)
]

prev_time=time.time()

while 1:
    is_true,frame=capture.read()
    frame=cv2.flip(frame,1)
    if not is_true:
        continue

    h,w,_=frame.shape
    frame_rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    mp_image=mp.Image(image_format=mp.ImageFormat.SRGB,data=frame_rgb)
    res=detector.detect(mp_image)

    if len(res.hand_landmarks)>0:
        l=[]

        for i in range(21):
            pos=res.hand_landmarks[0][i]
            x=int(pos.x*w)
            y=int(pos.y*h)
            l.append((x,y))

            cv2.circle(frame,(x,y),5,(0,255,0),-1)
            cv2.putText(frame,str(i),(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,0,255),1)

        for start,end in connections:
            cv2.line(frame,l[start],l[end],(255,0,0),2)

    current_time=time.time()
    fps=int(1/(current_time-prev_time))
    prev_time=current_time

    cv2.putText(frame,f"FPS: {fps}",(20,40),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
    cv2.imshow("video",frame)

    if cv2.waitKey(20)&0xFF==ord("d"):
        break
    elif cv2.waitKey(20)&0xFF==ord("p"):
        if len(res.hand_landmarks)>0:
            l=[]
            for i in range(21):
                pos=res.hand_landmarks[0][i]
                x=pos.x
                y=pos.y
                l.extend([x,y])
            l.append("ok")
            with open("data\\data.csv","a",newline="") as f:
                writer=csv.writer(f)
                writer.writerow(l)
                print('done')
capture.release()
cv2.destroyAllWindows()