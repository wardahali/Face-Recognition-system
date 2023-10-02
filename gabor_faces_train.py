import os
from PIL import Image
import numpy as np
import cv2
import pandas as pd
import functions as f


#dir=os.listdir()
#print(dir)

face_cascade= cv2.CascadeClassifier('cascades\data\haarcascade_frontalface_alt2.xml')

y_labels=[]
x_train=[]

current_id=0
label_ids={}


paths=r"D:\University\6th semester\Machine Learning\Project\Data minimized"
count=0
df_all=pd.DataFrame()
for root, dirs, files in os.walk(paths):
    for file in files:
        if file.endswith("jpg"):
            
            path= os.path.join(root,file)
            label= os.path.basename(os.path.dirname(path))
            
            #converting image to grayscale
            pil_image = Image.open(path).convert("L")
            pil_image=np.array(pil_image, "uint8")
           
            #
            faces = face_cascade.detectMultiScale(pil_image, 1.1, 4)
            
            for (x, y, w, h) in faces:
                #drawing rectangle on the face, start cordinates, end-cordinates, color, stroke
                cv2.rectangle(faces, (x, y), (x+w, y+h), (0, 0, 255), 2)
                img = pil_image[y:y + h, x:x + w]
                
                #compressing images from 1080 to 256
                image_array=cv2.resize(img,(256,256))
                
                filtered_img = f.gabor(image_array,5)
                #filtered_img = f.gabor(image_array,9)
                #filtered_img = f.hog(image_array)
                
                #storing image in dataframe
                df=pd.DataFrame(filtered_img)
                print(df.shape)
              

            
            df['Label'] = label
            
            count=count+1
            print(count)
            #merging all images in one dataframes
            df_all=pd.concat([df_all, df], ignore_index=True)
            print(df_all.shape)
        #converting dataframe to pickle
        df_all.to_pickle(r'D:\University\6th semester\Machine Learning\Project\Project Data\GaborData_ksize9test.pkl')
    
