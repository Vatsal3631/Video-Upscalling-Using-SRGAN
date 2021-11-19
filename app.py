# importing necessary libraries and functions
import numpy as np
from flask import Flask, request, jsonify, render_template, send_file, redirect, url_for
from sr_class import SR
from werkzeug.utils import secure_filename
import os
import cv2
import imquality.brisque as brisque
import tensorflow as tf
from os.path import isdir
import shutil
from tensorflow.keras.preprocessing.image import array_to_img
from tensorflow.keras.preprocessing.image import save_img
from flask_share import Share
share = Share()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
port = int(os.environ.get("PORT", 5000))

app = Flask(__name__) #Initialize the flask App
share.init_app(app)
#model = pickle.load(open('model.pkl', 'rb')) # loading the trained model


@app.route('/') # Homepage
def home():
    return render_template('index.html')
    
UPLOAD_FOLDER = 'uploads/'    
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
  
    
@app.route('/upScale',methods=['Get','POST'])
def upScale():
    file = request.files['file_name']
    quality = request.form['quality']
    if quality == "Base":
        scale = request.form['scale']
    else: 
        scale = 1
    if file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        size, fs_img, Final_Img = upScale_image(file,quality,scale)
        image = cv2.imread(app.config['UPLOAD_FOLDER']+"/Image.jpg")
        image = cv2.resize(np.asarray(image), size)
        acc0 = brisque.score(image)
        #fs_img = cv2.imread("./static/downloads/enhanced_photo01.jpg")
        acc1 = acc0 - brisque.score(fs_img) 
        acc2 = acc0 - brisque.score(Final_Img)
        return render_template('index1.html',prediction_text=round(acc1,4),prediction_text1=round(acc2,4))
    elif file.filename.lower().endswith(('.3gp', '.mp4', '.avi', '.mkv')):
        size = upScale_video(file,quality,scale)
        #cap = cv2.VideoCapture(app.config['UPLOAD_FOLDER']+"/Video.mp4")
        acc0 = frame_generator(app.config['UPLOAD_FOLDER']+"/Video.mp4",name='image000.jpg',size=size,save_img=False)
        #cap.release()
        #cap = cv2.VideoCapture("./static/downloads/fsrcnn_output.avi")
        score = frame_generator("./static/downloads/fsrcnn_output.avi",name='fsrcnn_image.jpg',size=size,save_img=True)
        acc1 = acc0 - score
        #cap.release()
        #cap = cv2.VideoCapture("./static/downloads/srgan_output.avi")
        score = frame_generator("./static/downloads/srgan_output.avi",name='srgan_image.jpg',size=size,save_img=True)
        acc2 = acc0 - score
        #cap.release()
        return render_template('index2.html',prediction_text=round(acc1,4),prediction_text1=round(acc2,4))

def frame_generator(path,name,size,save_img):
    cap = cv2.VideoCapture(path)
    ret,frame=cap.read()
    image = cv2.resize(frame, size)
    if save_img:
        path = 'static/downloads/'
        cv2.imwrite(os.path.join(path , name), image)
    cap.release()
    return brisque.score(image)

def upScale_video(vid,quality,scale):
    vid.save(os.path.join(app.config['UPLOAD_FOLDER'], "Video.mp4"))
    cam = cv2.VideoCapture(app.config['UPLOAD_FOLDER']+"/Video.mp4") 
    #fps = cam.get(cv2.CAP_PROP_FPS)
    if quality == "144p":
        size = (256,144)
    elif quality == "240p":
        size = (426,240)
    elif quality == "360p":
        size = (480,360)
    elif quality == "480p":
        size = (640,480)
    sr = SR()
    s_res = sr.res_video(cam) 
    frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH) * int(scale))
    frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT) * int(scale)) #int(cap.get(4))
    #height, width, layers = img.shape
    s1 = (frame_width,frame_height)
    if quality == "Base":
        sr.fsr_video(size=s1)
        size = s1
    else:
        sr.fsr_video(size=size)
        size = size    
    fps = 20       
    out = cv2.VideoWriter('./static/downloads/srgan_output.avi',cv2.VideoWriter_fourcc(*'MJPG'), fps , size)
    for i in range(len(s_res)):
        out.write(cv2.resize(cv2.imread(s_res[i]), size))
    out.release()
    return size

def upScale_image(img,quality,scale):
    '''
    For rendering results on HTML GUI
    '''
    sr = SR()
    sr.init_super("FSRCNN_x4")
    model = sr.srgan()   #generator()
    model.load_weights("gan_generator.h5")
    filename = secure_filename(img.filename)
    img.save(os.path.join(app.config['UPLOAD_FOLDER'], "Image.jpg"))
    if quality == "144p":
        size = (256,144)
    elif quality == "240p":
        size = (426,240)
    elif quality == "360p":
        size = (480,360)
    elif quality == "480p":
        size = (640,480)
    image = cv2.imread(app.config['UPLOAD_FOLDER']+"/Image.jpg")
    width = int(image.shape[1] * int(scale))
    height = int(image.shape[0] * int(scale))
    print(image.shape[1])
    s1 = (width,height)
    sres = sr.resolve_single(model, image)
    if quality == "Base":
        fs_img = sr.super_res(image, name= 'enhanced_photo01.jpg', save_img=True, scale=True, size=s1)
        Final_Img = cv2.resize(np.asarray(sres),s1)
        size = s1
    else:
        fs_img = sr.super_res(image, name= 'enhanced_photo01.jpg', save_img=True, scale=False, size=size)
        Final_Img = cv2.resize(np.asarray(sres),size)
        size = size
            
    path = 'static/downloads/enhanced_photo.jpg'
    cv2.imwrite(path, Final_Img)
    return size, fs_img, Final_Img
    

@app.route('/static/downloads')
def downloads():
    filename = "./static/downloads/enhanced_photo.jpg"
    return send_file(filename, as_attachment=True)

@app.route('/static')    
def download_video():
    filename = "./static/downloads/srgan_output.avi"
    return send_file(filename, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True,host = '0.0.0.0',port=port)
