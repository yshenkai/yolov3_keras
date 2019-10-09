import os
import colorsys
from timeit import default_timer as timer
import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image,ImageFont,ImageDraw
from model.model import yolo_body,yolo_eval
from utils.util import letterbox_image
class YOLO(object):
    _defaults={
        "model_path":"yolo.h5",
        "anchors_path":"data/yolo_anchors.txt",
        "classes_path":"data/voc_classes.txt",
        "score":0.15,
        "iou":0.2,
        "model_image_size":(416,416),
        "gpu_num":1
    }
    @classmethod
    def get_default(cls,n):
        if n in cls._defaults[n]:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '"+n+"'"
    def _get_classes(self):
        class_path=os.path.expanduser(self.classes_path)
        with open(class_path) as f:
            classes_name=f.readlines()
        classes_name=[c.strip() for c in classes_name]
        return classes_name
    def _get_anchors(self):
        anchors_path=os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors=f.readline()
        anchors=[float(x) for x in anchors.split(",")]
        anchors=np.array(anchors).reshape((-1,2))
        return anchors
    def generate(self):
        num_anchors=len(self.anchors)
        num_classes=len(self.classes_names)
        print("num_anchors",num_anchors)
        print("num_classes",num_classes)
        print("anchors",self.anchors)
        # print("classes",self.classes)
        self.yolo_model=yolo_body(Input(shape=(None,None,3)),num_anchors//3,num_classes)
        self.yolo_model.summary()
        self.yolo_model.load_weights(self.model_path)
        # self.yolo_model=load_model(self.model_path,compile=False)
        print("model loaded...")


        "生成边框颜色"
        hsv_tuples=[(x/len(self.classes_names),1.,1.) for x in range(len(self.classes_names))]
        self.colors=list(map(lambda x:colorsys.hsv_to_rgb(*x),hsv_tuples))
        self.colors=list(map(lambda x:(int(x[0]*255),int(x[1]*255),int(x[2]*255)),self.colors))
        np.random.seed(10101)
        np.random.shuffle(self.colors)
        np.random.seed(None)

        self.input_image_shape=K.placeholder(shape=(2,))
        print(self.yolo_model.output[0].shape,self.yolo_model.output[1].shape,self.yolo_model.output[2].shape)
        boxes,score,classes=yolo_eval(self.yolo_model.output,self.anchors,num_classes,self.input_image_shape,score_threshold=self.score,iou_threshold=self.iou)
        return boxes,score,classes
    def __init__(self,**kwargs):
        self.__dict__.update(self._defaults)
        self.__dict__.update(kwargs)
        self.classes_names=self._get_classes()
        self.anchors=self._get_anchors()
        self.sess=K.get_session()
        self.box,self.score,self.classes=self.generate()

    def detect_image(self,image):
        start=timer()
        if self.model_image_size!=(None,None):
            assert self.model_image_size[0]%32==0,"Multiples of 32 required"
            assert self.model_image_size[1]%32==0,"Multiples of 32 required"
            boxes_image=letterbox_image(image,tuple(reversed(self.model_image_size)))
        else:
            new_image_size=(image.width-(image.width%32),image.height-(image.height%32))
            boxes_image=letterbox_image(image,new_image_size)
        image_data=np.array(boxes_image,dtype="float32")
        print(image_data.shape)
        image_data=image_data/255.
        image_data=np.expand_dims(image_data,0)


        #********
        out_boxes,out_score,out_classes=self.sess.run([self.box,self.score,self.classes],feed_dict={
            self.yolo_model.input:image_data,
            self.input_image_shape:[image.size[1],image.size[0]],
            K.learning_phase():0
        })
        #********

        print("find {} boxes for {}".format(len(out_boxes),"img"))

        font=ImageFont.truetype(font="font/FiraMono-Medium.otf",size=np.floor(3e-2*image.size[1]+0.5).astype("int32"))
        thickness=(image.size[0]+image.size[1])//300

        for i,c in reversed(list(enumerate(out_classes))):
            predicted_class=self.classes_names[c]
            box=out_boxes[i]
            score=out_score[i]
            label="{} {:.2f}".format(predicted_class,score)
            draw=ImageDraw.Draw(image)
            label_size=draw.textsize(label,font)
            top,left,bottom,right=box
            top=max(0,np.floor(top+0.5).astype("int32"))
            left=max(0,np.floor(left+0.5).astype("int32"))
            bottom=min(image.size[1],np.floor(bottom+0.5).astype("int32"))
            right=min(image.size[0],np.floor(right+0.5).astype("int32"))
            print(label,(left,top),(right,bottom))
            if(top-label_size[1])>=0:
                text_orign=np.array([left,top-label_size[1]])
            else:
                text_orign=np.array([left,top+1])
            for i in range(thickness):
                draw.rectangle([left+i,top+i,right-i,bottom-i],outline=self.colors[c])
            draw.rectangle([tuple(text_orign),tuple(text_orign+label_size)],fill=self.colors[c])
            draw.text(text_orign,label,fill=(0,0,0),font=font)
            del draw
        end=timer()
        print(end-start)
        image.show()
        return image
    def close_session(self):
        self.sess.close()

if __name__=="__main__":
    yolo=YOLO()
    image=Image.open("2007_000175.jpg")
    image.show()
    res_image=yolo.detect_image(image)
