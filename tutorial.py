import torch, matplotlib.pyplot as plt  
import numpy as np  

import cv2 , uuid,os ,time, ssl , certifi
from urllib import request

ssl._create_default_https_context = ssl._create_unverified_context

model = torch.hub.load('ultralytics/yolov5', 'custom', path = 'runs/train/exp27/weights/last.pt', force_reload=True)

# img  = os.path.join('data', 'images', 'awake.0a506b94-a41b-11ee-8ef2-f21a94e3088d.jpg')
# results =  model(img)
# results.print()

# %matplotlib inline
# plt.imshow(np.squeeze(results.render()))
# plt.show()
# IMAGES_PATH = os.path.join('data', 'images')
# labels = ['awake', 'drowsy']
# number_imgs = 20


cap =  cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    results = model(frame)
    
    cv2.imshow('YOLOv5', np.squeeze(results.render()))
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

# for label in labels:
#     print('Collecting images for {}'.format(labels))
#     time.sleep(5)
    
#     for img_num in range(number_imgs):
#         print('Collecting images for {}, image number{}'.format(label, img_num))
#         #webcam feed
#         ret, frame = cap.read()


#         #Naming image path
#         imgname = os.path.join(IMAGES_PATH, label+'.'+str(uuid.uuid1())+'.jpg')

#         #writing images to file
#         cv2.imwrite(imgname, frame)

#         #render to screen
#         cv2.imshow('Image Collection', frame)

#         #delay between captures
#         time.sleep(2)

#         if cv2.waitKey(10) & 0xFF == ord('q'):
#             break
# cap.release()
# cv2.destroyAllWindows()
