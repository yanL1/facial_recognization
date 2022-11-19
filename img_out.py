import os, glob
from imutils import build_montages
import cv2

face_folder = os.walk(os.path.join("output"))

for path, dir_list, file_list in face_folder:
    faces = []
    if file_list:
        for file in file_list:
            image = cv2.imread(path + "\\" + file)
            faces.append(image)
    if faces:
        montage = build_montages(faces, (96, 96), (5, 5))[0]
        title = "Face ID #{}".format(path)
        cv2.imshow(title, montage)
        cv2.waitKey(0)
