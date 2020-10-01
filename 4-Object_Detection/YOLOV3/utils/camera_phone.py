# import requests
# import numpy as np
# import cv2
# while True:
#     img_res = requests.get("http://10.0.0.102:8080")
#     img_arr = np.array(bytearray(img_res.content), dtype = np.uint8)
#     img = cv2.imdecode(img_arr,-1)

#     cv2.imshow('frame', img)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break


import cv2
url = "http://10.0.0.102:8080" # Your url might be different, check the app
vs = cv2.VideoCapture(url+"/video")

while True:
    ret, frame = vs.read()
    if not ret:
        continue
    # Processing of image and other stuff here
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break