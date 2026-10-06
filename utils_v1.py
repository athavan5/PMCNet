from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import os
from PIL import Image
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import numpy as np 
from matplotlib import pyplot as plt
import cv2
from sklearn.metrics import *
from tensorflow.keras.utils import to_categorical

class Evaluator:
    def __init__(self):
        self.Dice = list()       
        self.IoU_polyp = list()

    def evaluate(self, pred, gt):
        
        pred_binary = pred
        pred_binary_inverse = (pred_binary == 0)

        gt_binary = (gt >= 0.5)
        gt_binary_inverse = (gt_binary == 0)
        
        TP = (pred_binary*gt_binary).sum()
        FP = (pred_binary*gt_binary_inverse).sum()
        TN = (pred_binary_inverse*gt_binary_inverse).sum()
        FN = (pred_binary_inverse*gt_binary).sum()

        if TP == 0:
            TP = 0.1
        # recall
        Recall = TP / (TP + FN)
        # Precision or positive predictive value
        Precision = TP / (TP + FP)
        #Specificity = TN / (TN + FP)
        # F1 score = Dice
        Dice = 2 * Precision * Recall / (Precision + Recall)
        # Overall accuracy
        Accuracy = (TP + TN) / (TP + FP + FN + TN)
        # IoU for poly
        IoU_polyp = TP / (TP + FP + FN)

        return  Dice, IoU_polyp

        
    def update(self, pred, gt):
        dice, ioU_polyp = self.evaluate(pred, gt)        
        self.Dice.append(dice)       
        self.IoU_polyp.append(ioU_polyp)

    def show(self):
        
        return round(np.mean(self.Dice)*100,2),round(np.mean(self.IoU_polyp)*100,2)



def resize_label(masks,w,h):
    new_mask = []
    for i in range(len(masks)):
        label = cv2.resize(masks[i],(w,h),interpolation = cv2.INTER_NEAREST)
        new_mask.append(label)
    return np.array(new_mask)
    
def make_label(label, num_class=5):
    new = to_categorical(label, num_class) # NOTE: changed from 3 to num_class default is 5 for IDRiD
    return new
   
def dataGenerator(batch_size,target_size,train_path,image_folder='images',mask_folder='masks',
                  seed = 100,image_color_mode = "rgb",mask_color_mode = "grayscale"):
    
    image_datagen = ImageDataGenerator(fill_mode='nearest')
    mask_datagen = ImageDataGenerator(fill_mode='nearest')
    
    image_generator = image_datagen.flow_from_directory(train_path,classes = [image_folder],class_mode = None,
                       color_mode = image_color_mode,target_size = target_size,batch_size = batch_size,seed = seed)
    
    mask_generator = mask_datagen.flow_from_directory(train_path,classes = [mask_folder], class_mode = None,
                       color_mode = mask_color_mode,target_size = target_size,batch_size = batch_size,seed = seed)
                       
    train_generator = zip(image_generator, mask_generator)
    for (img,mask) in train_generator:
        labels = make_label(mask[:,:,:,0])
        yield (img,labels)
                
def get_full_test_data(imgs_dir,label_dir,h,w):
    iter_tot=0
    image_names = os.listdir(imgs_dir)
    image_names = sorted(image_names)
    images_num  = len(image_names)  
    images  = np.empty((images_num,h,w,3))
    labels  = np.empty((images_num,h,w))
    
    for name in (image_names): 
            image = Image.open(imgs_dir+name)
            #image = image.resize((h,h))
            image = np.asarray(image) 
            image = cv2.resize(image,(w,h),interpolation = cv2.INTER_NEAREST)
            label = Image.open(label_dir + name[:-4]+'.png')
            #label = label.resize((h,h))
            label = np.asarray(label)
            if label.ndim == 3:
                # Some datasets store masks as RGB/RGBA images.
                # Accept only grayscale-equivalent masks where all color channels match.
                if not np.array_equal(label[:, :, 0], label[:, :, 1]) or not np.array_equal(label[:, :, 0], label[:, :, 2]):
                    raise ValueError(
                        f"Non-identical RGB channels in label mask: {label_dir + name[:-4] + '.png'}"
                    )
                label = label[:, :, 0]
            label = cv2.resize(label,(w,h),interpolation = cv2.INTER_NEAREST)
            images[iter_tot]=image
            labels[iter_tot]=label
            iter_tot=iter_tot+1
    return images, image_names, labels     
    
def get_scores(pred,label):
    y_pred = pred.flatten()
    y_true  = label.flatten()
    return y_pred, y_true 

def PR_AUC(pred, label):
    y_pred, y_true = get_scores(pred, label)
    if np.sum(y_true) == 0:
        # No positive samples: PR curve is undefined, return 0 to avoid NaN
        return 0.0
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    pr_auc = auc(recall, precision)
    return pr_auc if np.isfinite(pr_auc) else 0.0
    
def save_results(images, image_names, probs, lables, save_dir, h, w):
    os.makedirs(save_dir, exist_ok=True)
    predictions = np.argmax(probs,axis=-1)
    test_num = len(images)
    for i, name in enumerate(image_names):
        gray_image=images[i]
        label = lables[i]
        pred = predictions[i]
        
        label_vis = np.zeros((h,w,3),np.uint8)
        label_vis[label==1]=[255,0,0] # Red
        label_vis[label==2]=[0,255,0] # Green
        label_vis[label==3]=[0,0,255] # Blue
        label_vis[label==4]=[255,0,255] # Magenta
        
        pred_vis = np.zeros((h,w,3),np.uint8)
        pred_vis[pred==1]=[255,0,0]
        pred_vis[pred==2]=[0,255,0]
        pred_vis[pred==3]=[0,0,255]
        pred_vis[pred==4]=[255,0,255]
        cv2.imwrite(os.path.join(save_dir, os.path.splitext(os.path.basename(name))[0] + '_pred.png'), pred_vis[:,:,::-1])
    print(f"[INFO] Results saved to: {save_dir}")
        
        

def Multi_dataGenerator(batch_size,target_size,train_path,image_folder='images',mask_folder='masks',
                  seed = 1000,image_color_mode = "rgb",mask_color_mode = "grayscale"):
    w,h = 1440,960
    image_datagen = ImageDataGenerator(fill_mode='nearest')
    mask_datagen = ImageDataGenerator(fill_mode='nearest')
    
    image_generator = image_datagen.flow_from_directory(train_path,classes = [image_folder],class_mode = None,
                       color_mode = image_color_mode,target_size = target_size,batch_size = batch_size,seed = seed)
    
    mask_generator = mask_datagen.flow_from_directory(train_path,classes = [mask_folder], class_mode = None,
                       color_mode = mask_color_mode,target_size = target_size,batch_size = batch_size,seed = seed)
                       
    train_generator = zip(image_generator, mask_generator)
    for (img,mask) in train_generator:
        labels = make_label(mask[:,:,:,0])
        label_2  = resize_label(mask[:,:,:,0],w//2,h//2)
        label_4  = resize_label(mask[:,:,:,0],w//4,h//4)
        label_8  = resize_label(mask[:,:,:,0],w//8,h//8)
        label_16 = resize_label(mask[:,:,:,0],w//16,h//16)
        label_2 = make_label(label_2)
        label_4 = make_label(label_4)
        label_8 = make_label(label_8)
        label_16 = make_label(label_16)
        yield (img,[labels,label_2 ,label_4 ,label_8])
               
        
        