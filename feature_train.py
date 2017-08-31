# feature training
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from scipy.ndimage.measurements import label
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
from lesson_functions import *
import numpy as np
import pickle
import cv2
import glob
import time


images_car = glob.glob('vehicles/**/*.png')
images_noncar = glob.glob('non-vehicles/**/*.png')
print('cars:',len(images_car), '\n', 'non cars:', len(images_noncar))

plotOn = False

if plotOn == True:
    f, axs = plt.subplots(2,8, figsize=(10, 2))
    f.subplots_adjust(hspace = 0.3, wspace=0.1)
    axs = axs.ravel()
    for i in np.arange(8):
        img = cv2.imread(images_car[np.random.randint(0,len(images_car))])
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        axs[i].axis('off')
        axs[i].set_title('car', fontsize=10)
        axs[i].imshow(img)
    for i in np.arange(8,16):
        img = cv2.imread(images_noncar[np.random.randint(0,len(images_noncar))])
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        axs[i].axis('off')
        axs[i].set_title('non car', fontsize=10)
        axs[i].imshow(img)

# Feature extraction parameters
colorspace = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 11
pix_per_cell = 16
cell_per_block = 2
hog_channel = 'ALL'  # Can be 0, 1, 2, or "ALL"
hist_bins = 32
spatial_size = (32,32)
spatial_feat = True
hist_feat = False
hog_feat = True


t = time.time()

car_features = extract_features(images_car, color_space=colorspace,spatial_size=spatial_size,hist_bins=hist_bins, orient=orient, 
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat = spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)
notcar_features = extract_features(images_noncar, color_space=colorspace,spatial_size=spatial_size,hist_bins=hist_bins, orient=orient, 
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat = spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to extract HOG features...')
# Create an array stack of feature vectors
X = np.vstack((car_features, notcar_features)).astype(np.float64)  

# Fit a per-column scaler - this will be necessary if combining different types of features (HOG + color_hist/bin_spatial)
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))


# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)

print('Using:',orient,'orientations',pix_per_cell,
    'pixels per cell and', cell_per_block,'cells per block')
print('Feature vector length:', len(X_train[0]))

# Use a linear SVC 
svc = LinearSVC()
# Check the training time for the SVC
t = time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t=time.time()
n_predict = 10
print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
print('For these',n_predict, 'labels: ', y_test[0:n_predict])
t2 = time.time()
print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')

feature_parm = {'colorspace': colorspace,
                'orient': orient,
                'pix_per_cell': pix_per_cell,
                'cell_per_block': cell_per_block,
                'hog_channel': hog_channel, 
                'hist_bins': hist_bins,
                'spatial_size':spatial_size,
                'spatial_feat': spatial_feat,
                'hist_feat': hist_feat,
                'hog_feat': hog_feat,
                'svc':svc,
                'X_scaler': X_scaler,
                }
pkl_file = open('feature_parm.pkl', 'wb')
pickle.dump(feature_parm, pkl_file)