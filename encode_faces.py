# import the necessary packages
from imutils import paths
import face_recognition
import pickle
import cv2

dataset = "faces"
encoding_path = "encodings.pickle"
detection_method = "cnn"

print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images(dataset))
data = []

# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
    # load the input image and convert it from RGB (OpenCV ordering)
    # to dlib ordering (RGB)
    print("[INFO] processing image {}/{}".format(i + 1, len(imagePaths)))
    print(imagePath)

    try:
        image = cv2.imread(imagePath)
        # print(image.shape)  ->(449, 358, 3)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # detect the (x, y)-coordinates of the bounding boxes
        # corresponding to each face in the input image
        # 定位图像中的人脸
        boxes = face_recognition.face_locations(rgb, number_of_times_to_upsample=0, model=detection_method)
        if len(boxes) == 0:
            continue
        print(boxes)
        # compute the facial embedding for the face
        # 获取图像文件中所有面部编码
        encodings = face_recognition.face_encodings(rgb, boxes)
    except BaseException as e:
        print(e)
        continue

    # build a dictionary of the image path, bounding box location,
    # and facial encodings for the current image
    d = [{"imagePath": imagePath, "loc": box, "encoding": enc}
         for (box, enc) in zip(boxes, encodings)]
    data.extend(d)

# dump the facial encodings data to disk
print("[INFO] serializing encodings...")
f = open(encoding_path, "wb")
f.write(pickle.dumps(data))
f.close()
print('over')
