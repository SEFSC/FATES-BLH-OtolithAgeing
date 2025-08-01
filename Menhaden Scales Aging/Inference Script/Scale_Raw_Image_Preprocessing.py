import argparse
import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
import os
from os import listdir
from os.path import isfile, join
import numpy as np
from PIL import Image
import yaml
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import torch

def combine_masks(anns):
    img_area = anns[0]['segmentation'].shape[0]*anns[0]['segmentation'].shape[1]

    foreground_anns = []
    # remove any background masks
    for ann in anns:
        xmin,ymin,w,h = ann['bbox']
        if(w*h < 0.9*img_area):
            m = ann['segmentation']
            foreground_anns.append(ann)
    # keeps track of which masks should be deleted
    del_indices = [0]
    while(len(del_indices) > 0):
        del_indices = []
        #print(len(foreground_anns))
        for i in range(len(foreground_anns)):
            if(i not in del_indices):
                for j in range(i+1, len(foreground_anns)):
                    # check in bounding boxes overlap
                    if(j not in del_indices):
                        
                        left1,top1,w1,h1 = foreground_anns[i]['bbox']
                        left2,top2,w2,h2 = foreground_anns[j]['bbox']
                        right1 = left1+w1
                        right2 = left2+w2
                        bot1= top1+h1
                        bot2 = top2+h2
                        margin = 10
                        if(right1 >= left2-margin and left1 <= right2+margin and bot1 >= top2-margin and top1 <= bot2+margin):
                            #print(i,j)
                            foreground_anns[i]['segmentation'] = np.logical_or(foreground_anns[i]['segmentation'], foreground_anns[j]['segmentation'])
                            left = min(left1, left2)
                            right = max(right1,right2)
                            top = min(top1,top2)
                            bot = max(bot1,bot2)
                            foreground_anns[i]['bbox'] = [left, top, right-left, bot-top]
                            del_indices.append(j)
                            break
        combined_anns = []
        for i in range(len(foreground_anns)):
            if i not in del_indices:
                combined_anns.append(foreground_anns[i])
        foreground_anns = combined_anns
    return combined_anns

def choose_one(anns):
    scores= []

    # get max area and center of image
    h,w = anns[0]['segmentation'].shape[0], anns[0]['segmentation'].shape[1]
    centx =w/2
    centy= h/2
    max_area = 0
    for i in range (len(anns)):
        bbox = anns[i]['bbox']
        area = bbox[3]*bbox[2]
        if(area>max_area):
            max_area = area
    for i in range (len(anns)):
        score = 0
        anno = anns[i]
        bbox = anno["bbox"]
        # mask size score based on bbox area
        area_score =bbox[2]*bbox[3]/max_area
        # distance from center score (closer is better, so higher score)
        distance = (bbox[0]+bbox[2]*0.5-centx)**2 + (bbox[1]+bbox[3]*0.5-centy)**2
        distance_score = 1-(distance/(centy**2 + centx**2))
        score = area_score + distance_score
        scores.append(score)
    mask_index = scores.index(max(scores))
    return anns[mask_index]

def crop_and_pad_binary_threshold(image, kernel_size = 10, threshold=100, pad=0.05, bottom_pad = 0.35):
    #Binary threshold
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(gray_image,threshold,255,cv.THRESH_BINARY)
    kernel = np.ones((kernel_size,kernel_size),np.uint8)
    
    #Morphological operation
    opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel)
    closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel)
    
    #Find Contours
    contours, hierarchy = cv.findContours(closing, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    if(len(contours) > 0):
        maxidx = 0
        maxarea = 0
        for i in range(len(contours)):
            contour = contours[i]
            if cv.contourArea(contour) > maxarea:
                maxarea = cv.contourArea(contour)
                maxidx = i
        x,y,w,h = cv.boundingRect(contours[maxidx])
        
    #Crop and Pad
    xmargin = pad
    ymargin_top = pad
    ymargin_bot = bottom_pad
    y_max, x_max,c = image.shape
    x_new = int(max(0,x-w*xmargin))
    y_new = int(max(0,y-h*ymargin_top))
    w_new = int(min(w*(1+xmargin*2),x_max-x_new))
    h_new = int(min(h*(1+ymargin_top+ymargin_bot), y_max-y_new))
    crop = image[y_new:y_new+h_new,x_new:x_new+w_new,:]
    
    c_h, c_w, c = crop.shape
    left, right, top, bottom = 0,0,0,0
    if(c_h>c_w):
        difference = c_h-c_w
        left = int(difference/2)
        right = int(difference/2)
    if(c_h<c_w):
        difference = c_w-c_h
        top = int(difference/2)
        bottom = int(difference/2)
    crop_pad = cv.copyMakeBorder(crop, top, bottom, left, right, cv.BORDER_REPLICATE)

    return crop_pad

def crop_and_pad_sam(image, down_scale, sam_type, sam_model_path, num_points, thresh, pad, bottom_pad):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sam = sam_model_registry[sam_type](checkpoint=sam_model_path)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=num_points,
        stability_score_thresh=thresh,
    )
    resized_image = cv.resize(image, (0,0), fx=down_scale, fy=down_scale)
    masks = mask_generator.generate(resized_image)
    if(len(masks) == 0):
        return image
    combined_masks = combine_masks(masks)
    if(len(combined_masks) == 0):
        return image
    mask = choose_one(combined_masks)
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    xmargin = pad
    ymargin_top = pad
    ymargin_bot = bottom_pad
    y_max, x_max = gray_image.shape
    x,y,w,h = mask['bbox']
    x_new = int(max(0,x-w*xmargin)/down_scale)
    y_new = int(max(0,y-h*ymargin_top)/down_scale)
    w_new = int(min(w*(1+xmargin*2)/down_scale,x_max-x_new))
    h_new = int(min(h*(1+ymargin_top+ymargin_bot)/down_scale, y_max-y_new))
    crop = image[y_new:y_new+h_new,x_new:x_new+w_new,:]
    
    c_h, c_w, c = crop.shape
    left, right, top, bottom = 0,0,0,0
    if(c_h>c_w):
        difference = c_h-c_w
        left = int(difference/2)
        right = int(difference/2)
    if(c_h<c_w):
        difference = c_w-c_h
        top = int(difference/2)
        bottom = int(difference/2)
    crop_pad = cv.copyMakeBorder(crop, top, bottom, left, right, cv.BORDER_REPLICATE)
    return crop_pad

def preprocess_folder(image_dir, output_dir, seg_opt="binary", extension = ".tif", out_type = ".jpg", kernel_size = 10, threshold=100, pad=0.05, bottom_pad = 0.35,
                      normalization = "none", invert = False, down_scale = 0.5, sam_type = "vit_h", sam_model = "", num_pts = 8 , thresh = 0.88):
    if(not os.path.exists(output_dir)):
        os.mkdir(output_dir)
    for file in os.listdir(image_dir):
        if file.endswith(extension):
            image = cv.imread(image_dir+"/"+file)
            if(seg_opt =="binary"):
                cropped_image = crop_and_pad_binary_threshold(image, kernel_size, threshold, pad, bottom_pad)
            elif(seg_opt =="sam"):
                cropped_image = crop_and_pad_sam(image, down_scale, sam_type, sam_model, num_pts, thresh, pad, bottom_pad)
            else:
                cropped_image = image

            if(normalization == "he"):
                cropped_image = cv.cvtColor(cropped_image, cv.COLOR_BGR2GRAY)
                if(invert):
                    cropped_image = 255-cropped_image
                cropped_image = cv.equalizeHist(cropped_image)
                cropped_image = cv.cvtColor(cropped_image, cv.COLOR_GRAY2BGR)
                
            elif(normalization =="clahe"):
                cropped_image = cv.cvtColor(cropped_image, cv.COLOR_BGR2GRAY)
                if(invert):
                    cropped_image = 255-cropped_image
                clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                cropped_image = clahe.apply(cropped_image)
                cropped_image = cv.cvtColor(cropped_image, cv.COLOR_GRAY2BGR)

            
            cv.imwrite(output_dir+"/"+os.path.splitext(file)[0]+out_type,cropped_image)

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", help="path to configuration yaml file")

args = parser.parse_args()

        
with open(args.config_path, 'r') as file:
    config = yaml.safe_load(file)

preprocess_folder(config["raw_img_pth"], config["preprocessed_img_pth"], config["segment"], config["input_type"], config["output_type"],10, config["binary_threshold"],
                  config["pad"], config["bottom_pad"], config["normalization"], config["invert"],
                  config["downsample"], config["sam_model_type"], config["sam_weights_path"], config["points_per_side"], config["stability_score_thresh"])

