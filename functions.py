# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 18:41:53 2021

@author: wardah
"""
import cv2
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog


def run_hog(image_array):
     hogfv, hog_image = hog(image_array, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2),visualize=True)
     return hogfv.reshape(1,-1)
 
def gabor(image_array, ksize):
    theta= 4.71238898038469
    sigma= 3
    lamda= 2.356194490192345
    gamma= 0.5
    phi = 0  
    image_array = cv2.equalizeHist(image_array)
    kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, phi, ktype=cv2.CV_32F)    
    
    #Now filter the image and add values to new column
    fimg = cv2.filter2D(image_array, cv2.CV_8UC3, kernel)
    
    return fimg.reshape(1,-1)       
                                         


def pca(X,n_samples, n_features):
    n_components = min(n_samples, n_features) - 1
    #applying standard scalar to feature vector
    x=StandardScaler().fit_transform(X)
    
    pca=PCA(n_components)
    #fiting the model and applying dimensionality reduction
    comp_features=pca.fit_transform(x)
    print(comp_features.shape)
    return comp_features