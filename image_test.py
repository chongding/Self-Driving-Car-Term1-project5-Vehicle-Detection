# pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from scipy.ndimage.measurements import label
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
from utilities import *
import numpy as np
import pickle
import cv2
import glob
import time


# load feature parameters
pkl_file = open('feature_parm.pkl', 'rb')
feature_parm = pickle.load(pkl_file)

colorspace = feature_parm['colorspace']
orient = feature_parm['orient']
pix_per_cell = feature_parm['pix_per_cell']
cell_per_block = feature_parm['cell_per_block']
hog_channel = feature_parm['hog_channel']
hist_bins = feature_parm['hist_bins']
spatial_size = feature_parm['spatial_size']
spatial_feat = feature_parm['spatial_feat']
hist_feat = feature_parm['hist_feat']
hog_feat = feature_parm['hog_feat']
svc = feature_parm['svc']
X_scaler = feature_parm['X_scaler']

img_count = 0
def vehicle_det(image):
    global last_bbboxs
    global img_count
    img_count += 1
    draw_image = np.copy(image)
    if img_count % 2 != 0:
        #find cars in frame
    #    windows1 = find_cars(draw_image, 400, 500, 0.8, svc, X_scaler,colorspace, spatial_size, hist_bins, orient, pix_per_cell, cell_per_block, hog_channel, spatial_feat=spatial_feat, hist_feat=hist_feat)
        windows2 = find_cars(draw_image, 400, 660, 1, svc, X_scaler,colorspace, spatial_size, hist_bins, orient, pix_per_cell, cell_per_block, hog_channel, spatial_feat=spatial_feat, hist_feat=hist_feat)
        windows3 = find_cars(draw_image, 400, 660, 1.5, svc, X_scaler,colorspace, spatial_size, hist_bins, orient, pix_per_cell, cell_per_block, hog_channel, spatial_feat=spatial_feat, hist_feat=hist_feat)
#        windows4 = find_cars(draw_image, 400, 660, 2, svc, X_scaler,colorspace, spatial_size, hist_bins, orient, pix_per_cell, cell_per_block, hog_channel, spatial_feat=spatial_feat, hist_feat=hist_feat)

        windows = windows2 + windows3 
        
        # Add heat to each box in box list
        heat = np.zeros_like(draw_image[:,:,0]).astype(np.float)
        heat = add_heat(heat,windows)
        # Apply threshold to help remove false positives
        heat = apply_threshold(heat,1)
        # Visualize the heatmap when displaying    
        heatmap = np.clip(heat, 0, 255)
        
        # Find final boxes from heatmap using label function
        labels = label(heatmap)
        bboxs = label2bboxs(labels)
        out_img = draw_boxes(np.copy(draw_image), bboxs)
        last_bboxs = bboxs
    else:
        out_img = draw_boxes(np.copy(draw_image), last_bboxs)
    return out_img, bboxs

test_image = mpimg.imread('./test_images/test1.jpg')
t = time.time()
out_img, bboxs = vehicle_det(test_image)
print(round(time.time()-t, 2), 'Seconds to process test images')
plt.imshow(out_img)








