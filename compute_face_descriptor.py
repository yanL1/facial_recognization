import os
import dlib
import glob
import cv2
import pickle
import time

# 指定路径
current_path = os.getcwd()
model_path = current_path + '/model/'
shape_predictor_model = model_path + '/shape_predictor_5_face_landmarks.dat'
face_rec_model = model_path + '/dlib_face_recognition_resnet_model_v1.dat'
face_folder = current_path + '/faces/'

# 导入模型
detector = dlib.get_frontal_face_detector()
shape_detector = dlib.shape_predictor(shape_predictor_model)
face_recognizer = dlib.face_recognition_model_v1(face_rec_model)

# 为后面操作方便，建了几个列表
data_set = []
# 遍历faces文件夹中所有的图片
start = int(round(time.time() * 1000))
for f in glob.glob(os.path.join(face_folder, "*")):
    print('Processing file：{}'.format(f))
    # 读取图片
    img = cv2.imread(f)
    # 转换到rgb颜色空间
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 检测人脸
    dets = detector(img2, 1)
    print("Number of faces detected: {}".format(len(dets)))

    # 遍历所有的人脸
    for index, face in enumerate(dets):
        # 检测人脸特征点
        shape = shape_detector(img2, face)
        # 投影到128D
        face_descriptor = face_recognizer.compute_face_descriptor(img2, shape)

        # 保存相关信息
        d = [{"image": f, "descriptor": face_descriptor}]
        data_set.extend(d)

end = int(round(time.time() * 1000))

# dump the facial encodings data to disk
print("[INFO] serializing encodings...")
f = open("encodings.pickle", "wb")
f.write(pickle.dumps(data_set))
f.close()
print('over')
print(end - start)