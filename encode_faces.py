# import the necessary packages
from imutils import paths
import face_recognition
import argparse
import pickle
import cv2
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", required=True,
                help="path to input directory of faces + images")
ap.add_argument("-e", "--encodings", required=True,
                help="path to serialized db of facial encodings")
ap.add_argument("-d", "--detection-method", type=str, default="hog",
                help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())

# grab the paths to the input images in our dataset, then initialize
# out data list (which we'll soon populate)
face_img_save_path = str(args['dataset']) + '/face_imgs/'
if not os.path.exists(face_img_save_path):
    os.makedirs(face_img_save_path)
# print(face_img_save_path)

print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images(args["dataset"]))
data = []

# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
    # load the input image and convert it from RGB (OpenCV ordering)
    # to dlib ordering (RGB)
    print("[INFO] processing image {}/{}".format(i + 1,
                                                 len(imagePaths)))
    print(imagePath)
    image = cv2.imread(imagePath)
    # print(image.shape)  ->(449, 358, 3)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # detect the (x, y)-coordinates of the bounding boxes
    # corresponding to each face in the input image
    # 定位图像中的人脸
    boxes = face_recognition.face_locations(rgb, model=args["detection_method"])
    if len(boxes) == 0:
        continue
    print(boxes)
    # print('识别人脸完成')
    # compute the facial embedding for the face
    # 获取图像文件中所有面部编码
    # print(face_img_save_path+imagePath.split('\\')[-1])
    # cv2.imwrite('here.jpg',boxes)
    for i in boxes:
        top, right, bottom, left = i
        face = rgb[top:bottom, left:right]
        print(face_img_save_path + imagePath.split('\\')[-1])
        cv2.imwrite(face_img_save_path + imagePath.split('\\')[-1], face)
    encodings = face_recognition.face_encodings(rgb, boxes)

    # build a dictionary of the image path, bounding box location,
    # and facial encodings for the current image
    d = [{"imagePath": imagePath, "loc": box, "encoding": enc}
         for (box, enc) in zip(boxes, encodings)]
    data.extend(d)

# dump the facial encodings data to disk
print("[INFO] serializing encodings...")
# print('data:',data)
f = open(args["encodings"], "wb")
f.write(pickle.dumps(data))
f.close()
print('over')
