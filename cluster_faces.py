# import the necessary packages
from sklearn.cluster import DBSCAN
from imutils import build_montages
import numpy as np
import argparse
import pickle
import cv2

encoding_path = "encodings.pickle"
jobs = -1

# load the serialized face encodings + bounding box locations from
# disk, then extract the set of encodings to so we can cluster on
# them
print("[INFO] loading encodings...")
data = pickle.loads(open(encoding_path, "rb").read())
data = np.array(data)
encodings = [d["encoding"] for d in data]
print('encodings.shape', np.shape(encodings))

# cluster the embeddings
print("[INFO] clustering...")
clt = DBSCAN(metric="euclidean", n_jobs=jobs)
# labels = clt.fit_predict(encodings)
# print(labels)
clt.fit(encodings)
print('clt.labels_:', clt.labels_)

# determine the total number of unique faces found in the dataset
labelIDs = np.unique(clt.labels_)
print('labelIDs.shape', np.shape(labelIDs))
print('clt.labels_.shape:', np.shape(clt.labels_))
numUniqueFaces = len(np.where(labelIDs > -1)[0])
print("[INFO] # unique faces: {}".format(numUniqueFaces))

# loop over the unique face integers
show_num = 30  # 显示人脸数量
for labelID in labelIDs:
    # find all indexes into the `data` array that belong to the
    # current label ID, then randomly sample a maximum of 25 indexes
    # from the set
    print("[INFO] faces for face ID: {}".format(labelID))
    idxs = np.where(clt.labels_ == labelID)[0]
    # print('idxs1',idxs)
    idxs = np.random.choice(idxs, size=min(show_num, len(idxs)), replace=False)
    # print('idxs',idxs)

    # initialize the list of faces to include in the montage
    faces = []

    # loop over the sampled indexes
    for i in idxs:
        # load the input image and extract the face ROI
        image = cv2.imread(data[i]["imagePath"])
        (top, right, bottom, left) = data[i]["loc"]
        face = image[top:bottom, left:right]

        # force resize the face ROI to 96x96 and then add it to the
        # faces montage list
        face = cv2.resize(image, (96, 96))
        faces.append(face)

    # create a montage using 96x96 "tiles" with 5 rows and 5 columns
    montage = build_montages(faces, (96, 96), (5, 5))[0]
    #
    # # show the output montage
    title = "Face ID #{}".format(labelID)
    title = "Unknown Faces" if labelID == -1 else title
    cv2.imshow(title, montage)
    cv2.waitKey(0)