import numpy as np
import keras.backend as K
from keras.layers import Input,Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard,ModelCheckpoint,ReduceLROnPlateau,EarlyStopping
from data.data_utils import data_generator_wrapper,get_anchors,get_classes,data_generator
from model.model import yolo_body,yolo_loss

def create_model(input_shape,anchors,num_classes,load_pretrained=True,freeze_body=2,weights_path="../data/yolo_weights1.h5"):
    K.clear_session()
    image_imput=Input(shape=(None,None,3))
    h,w=input_shape
    num_anchors=len(anchors)
    y_true=[Input(shape=(h//{0:32, 1:16, 2:8}[l], w//{0:32, 1:16, 2:8}[l],
        num_anchors//3, num_classes+5)) for l in range(3)]
    model_body=yolo_body(image_imput,num_anchors//3,num_classes)
    print("Created YOLOV3 model")
    if load_pretrained:
        model_body.load_weights(weights_path,by_name=True,skip_mismatch=True)
        print("load weight...")
        if freeze_body in [1,2]:
            num=(185,len(model_body.layers)-3)[freeze_body-1]
            for i in range(num):
                model_body.layers[i].trainable=False
            print("free model layer")
    model_loss=Lambda(yolo_loss,output_shape=(1,),name="yolo_loss",
                      arguments={"anchors":anchors,"num_classes":num_classes,"ignore_thresh":0.5})(
        [*model_body.output,*y_true])
    model=Model([model_body.input,*y_true],model_loss)
    return model


def _main():
    annotation_path_train="../data/train.txt"
    annotation_path_val="../data/2012_val.txt"
    log_dir="logs/000/"
    classes_path="../data/voc_classes.txt"
    anchors_path="../data/yolo_anchors.txt"
    classes=get_classes(classes_path)
    anchors=get_anchors(anchors_path)
    print("....",anchors)
    print(len(classes))
    num_classes=len(classes)
    input_shape=(416,416)
    model=create_model(input_shape,anchors,num_classes,load_pretrained=True)
    logging=TensorBoard(log_dir=log_dir)
    checkpoint=ModelCheckpoint(log_dir+"ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5",monitor="val_loss",save_weights_only=True,save_best_only=True,period=3)
    reduce_lr=ReduceLROnPlateau(monitor="val_loss",factor=0.1,patience=3,verbose=1)
    early_stopping=EarlyStopping(monitor="val_loss",min_delta=0,patience=10,verbose=1)
    val_split=0.1
    with open(annotation_path_train) as f:
        train_lines=f.readlines()
    np.random.seed(10101)
    np.random.shuffle(train_lines)
    np.random.seed(None)

    num_val=int(len(train_lines)*val_split)
    num_train=len(train_lines)-num_val

    model.compile(optimizer=Adam(lr=1e-3),loss={"yolo_loss":lambda y_true,y_pred:y_pred})
    model.summary()
    batch_size=8
    model.fit_generator(data_generator_wrapper(train_lines[:num_train],batch_size,input_shape,anchors,num_classes),
                        steps_per_epoch=max(1,num_train//batch_size),
                        validation_data=data_generator_wrapper(train_lines[num_train:],batch_size,input_shape,anchors,num_classes),
                        validation_steps=max(1,num_val//batch_size),
                        epochs=50,initial_epoch=0,callbacks=[logging,checkpoint])
    model.save_weights(log_dir+"trained_weight_stage_1.h5")


    model.load_weights(log_dir+"trained_weight_stage_1.h5")
    for i in range(len(model.layers)):
        model.layers[i].trainable = True
        model.compile(optimizer=Adam(lr=1e-4), loss={'yolo_loss': lambda y_true, y_pred: y_pred}) # recompile to apply the change
        print('Unfreeze all of the layers.')

        batch_size = 8 # note that more GPU memory is required after unfreezing the body
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(data_generator_wrapper(train_lines[:num_train], batch_size, input_shape, anchors, num_classes),
            steps_per_epoch=max(1, num_train//batch_size),
            validation_data=data_generator_wrapper(train_lines[num_train:], batch_size, input_shape, anchors, num_classes),
            validation_steps=max(1, num_val//batch_size),
            epochs=100,
            initial_epoch=50,
            callbacks=[logging, checkpoint, reduce_lr, early_stopping])
        model.save_weights(log_dir + 'trained_weights_final.h5')


if __name__=="__main__":
    _main()
