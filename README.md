# faster-rcnn
there are many files,but only a few of them will by useful for your training and evaluation.  
train_res50_fpn.py: run directly for training  
predict.py: run directly to use model to predict your sample  
eval_voc.py: run directly to get mAP on Voc testset of your model  
draw_proposal_box.py: run directly to get proposal boxes produced by the faster-rcnn model  
save_weights file: to save model trained from train_res50_fpn.py  

 before your train you should download these two models:  
 resnet50：https://download.pytorch.org/models/resnet50-0676ba61.pth  
 fasterrcnn pretraining models: https://download.pytorch.org/models/mobilenet_v2-b0353104.pth  
 put them into backbone file
