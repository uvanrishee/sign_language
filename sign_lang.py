import mediapipe as mp
import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import joblib
import numpy as np

model=joblib.load("objects\\model.pkl")
encoder=joblib.load("objects\\label_encoder.pkl")

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
        points=[]
        features=[]
        for i in range(21):
            pos=res.hand_landmarks[0][i]
            x=int(pos.x*w)
            y=int(pos.y*h)
            points.append((x,y))
            features.extend([pos.x,pos.y])
            cv2.circle(frame,(x,y),3,(0,255,0),-1)

        for start,end in connections:
            cv2.line(frame,points[start],points[end],(255,0,0),1)

        features=np.array(features).reshape(1,-1)

        pred=model.predict(features)
        probs=model.predict_proba(features)

        label=encoder.inverse_transform(pred)[0]
        confidence=np.max(probs)

        if confidence<0.7:
            label="UNKNOWN"

        confidence=int(confidence*100)

        cv2.putText(frame,label,(20,50),cv2.FONT_HERSHEY_DUPLEX,1.4,(0,0,0),5)
        cv2.putText(frame,label,(20,50),cv2.FONT_HERSHEY_DUPLEX,1.4,(0,255,0),2)

        cv2.putText(frame,f"Confidence: {confidence}%",(20,h-20),cv2.FONT_HERSHEY_DUPLEX,0.8,(0,0,0),4)
        cv2.putText(frame,f"Confidence: {confidence}%",(20,h-20),cv2.FONT_HERSHEY_DUPLEX,0.8,(255,255,255),1)

    cv2.imshow("video",frame)

    if cv2.waitKey(1)&0xFF==ord("d"):
        break

capture.release()
cv2.destroyAllWindows()