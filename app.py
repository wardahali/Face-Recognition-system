import tkinter as tk
from tkinter import *
import pandas as pd
import functions as f
import pickle
from tkinter import filedialog
import cv2
from PIL import ImageTk, Image
import numpy as np

root = Tk() #Tk provides a no. of widgets to develop desktop applications
            #root is the root window into which all other widgets go
root.title('FACE RECOGNITION SYSTEM')
root.configure(bg='gray') #background color
root.geometry("600x300+100+70") #decides the size and position of the root window

#Label is used to implement display boxes where you can place text, font style/size, colors
heading = tk.Label(root, text="FACE RECOGNITION SYSTEM", font= ("Times New Roman", 20), bg='gray', fg='black')
heading.pack() #packs widgets in rows or columns

def face_recognition(img):
        face_cascade= cv2.CascadeClassifier('cascades\data\haarcascade_frontalface_alt2.xml')   
        faces = face_cascade.detectMultiScale(img, 1.1, 1)
        
        if faces is ():
            return None
        
        # Crop all faces found
        for (x,y,w,h) in faces:

                image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)        #converting to grayscale
                image_array=cv2.resize(image,(256,256))
                #calling gabor function from functions.py
                gaborfv=f.gabor(image_array,5)
                #pcafv=f.pca(gaborfv,2691,655856)
                df=pd.DataFrame(gaborfv)
                loaded_model = pickle.load(open('model_gabor_linear.sav', 'rb'))
                predict=loaded_model.predict(df)
                
                #drawing rectangle on the face, starting cordinates, endinf cordinates, color, thickness
                cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=2)
            
                #displaying name of the face recognised in normal size serif font, font scale, color, thickness, line type
                return cv2.putText(img, predict[0], (x,y), cv2.FONT_HERSHEY_COMPLEX , 1, (0, 0, 255) , 2, cv2.LINE_AA)


def startcam():

    cap=cv2.VideoCapture(0)
  
    df=pd.DataFrame()
    while(True):
        #capture frame by frame
        ret,frame=cap.read()
      
        frame=face_recognition(frame)
        cv2.imshow('frame', frame)
        # ord converts character to int
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
#adds a button on the window, text with style/size, colors, assign the name of a function to the command option of the widget so that when the event occurs on the widget, the function will be called automatically.      
b1 = tk.Button(root, text="Real Time Evaluation",font= ("Times New Roman", 15), bg='black', fg='white', command=startcam)
b1.place(x=15, y=50) #Place allows you to explicitly set the position and size of a window


def Browse():

    global folder_path #Global variables are the one that is defined and declared outside a function and we need to use them inside a function.
    #Filedialog helps you open, save files or directories.
    filename = filedialog.askopenfilename() #askopenfilename() function returns the file name that you selected.
    folder_path.set(filename)
    load = Image.open(filename)
    
    # Load functions
    def faceee(img):
       
        # Function detects faces and returns the cropped face
        # If no face detected, it returns the input image
        

            img=face_recognition(img)
            cv2.imshow('Detected Faces', img)
        
            

    load=faceee(np.array(load, "uint8")) #loads the images by taking in an array
    
    
#global doesn't create variable but it is only used in function to inform this function to use external/global variable when you use = to assign value      
folder_path = StringVar() #creates a global variable

#adds a button on the window, text with style/size, colors, assign the name of a function to the command option of the widget so that when the event occurs on the widget, the function will be called automatically.
b2 = tk.Button(root, text="Image Person Identification",font= ("Times New Roman", 15), bg='black', fg='white', command=Browse)
b2.place(x=15, y=170) #Place allows you to explicitly set the position and size of a window
 

button_quit = tk.Button(root, text="Exit",font= ("Times New Roman", 10), bg='black', padx=22,
pady=5, fg='red', command=root.quit) #exits from the mainloop and all the widgets
button_quit.place(x=440, y=90) #Place allows you to explicitly set the position and size of a window

root.mainloop() #it will loop forever until the user exits the window