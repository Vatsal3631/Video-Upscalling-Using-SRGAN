import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from cv2 import dnn_superres
import tensorflow as tf
from utils import load_image, plot_sample
from os.path import isdir
import shutil
from tensorflow.keras.preprocessing.image import array_to_img
from tensorflow.keras.preprocessing.image import save_img
from PIL import Image
import tensorflow as tf
from tensorflow.python.keras.layers import Add, BatchNormalization, Conv2D, Dense, Flatten, Input, LeakyReLU, PReLU, Lambda
from tensorflow.python.keras.models import Model
from utils import load_image, plot_sample
#%cd Mini Project II/Final


#@app.route('/upscale',methods=['POST'])
#print(os.getcwd())
class SR:
    def init_super(self, model, base_path='Model'):
        global sr, model_name, model_scale      # Define global variable

        sr = dnn_superres.DnnSuperResImpl_create()   # Create an SR object

        model_path = os.path.join(model +".pb")    # Define model path

        model_name = model.split('_')[0].lower()     # Extract model name from model path

        model_scale = int(model.split("_")[1][1])    # Extract model scale from model path

        sr.readModel(model_path)    # Read the desired model

        sr.setModel(model_name, model_scale)

        
    
    def super_res(self, image, returndata=False, save_img=True, name='test.png', size=(256,144), scale=True):
        #image = cv2.imread("uploads/Image.jpg")
        #print(type(img))
        Final_Img = sr.upsample(image)     # Upscale the image
        if  returndata:
            return Final_Img

        else:

            if save_img:
                if scale:
                    Final_Img = cv2.resize(Final_Img,size)
                else:
                    Final_Img = cv2.resize(Final_Img,size)
                path = 'static/downloads/'
                cv2.imwrite(os.path.join(path , name), Final_Img)
                #cv2.imwrite("{{ url_for('predict'),filename=}}" + name, Final_Img)
        return Final_Img

    def fsr_video(self,size):
        fps=0
        self.init_super("FSRCNN_x4")
        cap = cv2.VideoCapture("./uploads/Video.mp4")
        size = size #(frame_width, frame_height) #(640,480)
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter('./static/downloads/fsrcnn_output.avi', fourcc, 20, size)

        while(cap.isOpened()):    
            
            ret,frame=cap.read() 
            
            if ret == True:
                image = cv2.flip(frame,180)
                
                image = cv2.flip(image,1)
            
                image = self.super_res(image, returndata=True, save_img=False, size=size) 
            
                #cv2.putText(image, 'FPS: {:.2f}'.format(fps), (10, 20), cv2.FONT_HERSHEY_SIMPLEX,0.8, (255, 20, 55), 1)
                #cv2.imshow("Super Resolution", image)
                
                out.write(cv2.resize(image, size))

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
            else:
                break

        cap.release() 
        out.release()
        cv2.destroyAllWindows()  
        
    def res_video(self,cam):
        try:
            if not os.path.exists('./uploads/data'): 
                os.makedirs('./uploads/data') 
        except OSError:
            print ('Error: Creating directory of data') 
      
        currentframe = 0
        arr_img = []
        while(True): 
            ret,frame = cam.read() 
            if ret: 
                name = './uploads/data/frame' + str(currentframe).zfill(3) + '.jpg'
                #print ('Creating...' + name) 
                cv2.imwrite(name, frame) 
                currentframe += 1
                arr_img.append(name)
            else: 
                break
        model = self.srgan()   #generator()
        model.load_weights('gan_generator.h5')

        arr_output=[]
        n= len(arr_img)

        for i in range(n):
            lr = load_image(arr_img[i])
            sres = self.resolve_single(model, lr)
            arr_output.append(sres)
        cam.release() 
        cv2.destroyAllWindows()
        
        if isdir("./static/downloads/output_images"):
            shutil.rmtree("./static/downloads/output_images")
        os.makedirs("./static/downloads/output_images")
        s_res= []
        for j in range(len(arr_output)):
            out_name = './static/downloads/output_images/frame' + str(j).zfill(3) + '.jpg'
            img_pil = array_to_img(arr_output[j])
            img1 = save_img(out_name, img_pil)
            s_res.append(out_name)
        return s_res
    
    def resolve_single(self,model, lr):
        return self.resolve(model, tf.expand_dims(lr, axis=0))[0]  
    
    def resolve(self,model, lr_batch):
        lr_batch = tf.cast(lr_batch, tf.float32)
        sr_batch = model(lr_batch)
        sr_batch = tf.clip_by_value(sr_batch, 0, 255)
        sr_batch = tf.round(sr_batch)
        sr_batch = tf.cast(sr_batch, tf.uint8)
        return sr_batch

    def pixel_shuffle(self,scale):
        return lambda x: tf.nn.depth_to_space(x, scale)

    def normalize_01(self,x):
        """Normalizes RGB images to [0, 1]."""
        return x / 255.0

    def denormalize_m11(self,x):
        """Inverse of normalize_m11."""
        return (x + 1) * 127.5

    def res_block(self,x_in, num_filters, momentum=0.8):
        x = Conv2D(num_filters, kernel_size=3, padding='same')(x_in)
        x = BatchNormalization(momentum=momentum)(x)
        x = PReLU(shared_axes=[1, 2])(x)
        x = Conv2D(num_filters, kernel_size=3, padding='same')(x)
        x = BatchNormalization(momentum=momentum)(x)
        x = Add()([x_in, x])
        return x

    def upsample(self,x_in, num_filters):
        x = Conv2D(num_filters, kernel_size=3, padding='same')(x_in)
        x = Lambda(self.pixel_shuffle(scale=2))(x)
        return PReLU(shared_axes=[1, 2])(x)

    def srgan(self,num_filters=64, num_res_blocks=16):
        x_in = Input(shape=(None, None, 3))
        x = Lambda(self.normalize_01)(x_in)

        x = Conv2D(num_filters, kernel_size=9, padding='same')(x)
        x = x_1 = PReLU(shared_axes=[1, 2])(x)

        for _ in range(num_res_blocks):
            x = self.res_block(x, num_filters)

        x = Conv2D(num_filters, kernel_size=3, padding='same')(x)
        x = BatchNormalization()(x)
        x = Add()([x_1, x])

        x = self.upsample(x, num_filters * 4)
        x = self.upsample(x, num_filters * 4)

        x = Conv2D(3, kernel_size=9, padding='same', activation='tanh')(x)
        x = Lambda(self.denormalize_m11)(x)

        return Model(x_in, x)
