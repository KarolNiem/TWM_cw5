import glob,os,cv2
from turtle import window_width
from imutils.object_detection import non_max_suppression
from skimage.feature import hog
from skimage import color
from skimage.transform import pyramid_gaussian
from sklearn.svm import LinearSVC
import numpy as np 

#Parameters
image_size=(64,128)
output_image_size=(400,256)
window_width=64
window_height=128
step=8
scale_step = 5/6
detections = []
Features = []
Labels = []
scale = 0
iou=0.5
positive_path=os.path.join('datasets/training/positive/*')
negative_path=os.path.join('datasets/training/negative/*')

#Positive images
for image in glob.glob(positive_path):
    cur_hog = hog(cv2.resize(cv2.imread(image,0),image_size))
    Features.append(cur_hog)
    Labels.append(1)
positive=len(Features)
print(f"Positive examples ({positive}) are loaded and labeled")

# Negative images
for image in glob.glob(negative_path):
    cur_hog = hog(cv2.resize(cv2.imread(image,0),image_size))
    Features.append(cur_hog)
    Labels.append(0)
negative=len(Features)-positive
print(f"Negative examples ({negative}) are loaded and labeled")

#SVM model
model = LinearSVC()
print("SVM model created")
model.fit(Features,Labels)
print("SVM model trained")
image = cv2.resize(cv2.imread('datasets/test/people_1.jpg'),output_image_size)

#Sliding window
for level in pyramid_gaussian(image, downscale = 1/scale_step):
    if window_height <= level.shape[0] and window_width <= level.shape[1]:
        for y in range(0, level.shape[0], step):
            for x in range(0, level.shape[1], step):
                window=level[y: y + window_height, x: x + window_width]
                if window.shape[0] == window_height and window.shape[1] == window_width:                    
                    cur_hog= hog(color.rgb2gray(window), orientations=9,pixels_per_cell=(8,8),visualize=False,cells_per_block=(3,3)).reshape(1,-1)
                    pred = model.predict(cur_hog)
                    if pred == 1 and model.decision_function(cur_hog) > iou:
                        cur_x = int(x * ((1/scale_step)**scale))
                        cur_y = int(y * ((1/scale_step)**scale))
                        cur_width = int(window_width * ((1/scale_step)**scale))
                        cur_height = int(window_height * ((1/scale_step)**scale))
                        detections.append((cur_x, cur_y, model.decision_function(cur_hog), cur_width,cur_height))
    
        scale += 1
    else:
        break

#Image display
clone = image.copy()
rects = np.array([[x, y, x + w, y + h] for (x, y, _, w, h) in detections])
sc = np.array([score[0] for (x, y, score, w, h) in detections])
for(x1, y1, x2, y2) in non_max_suppression(rects, probs = sc, overlapThresh = 0.3):
    cv2.rectangle(clone, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(clone,'Person',(x1,y1-5),1,1,(0, 255, 0),1)
cv2.imshow('Output image',clone)
cv2.waitKey(0)
cv2.destroyAllWindows()
