from ultralytics import YOLO
 
# Load the model.
# model = YOLO('yolov8l.pt')
model = YOLO("./runs/detect/240529_yolov8l-bbox_custom/weights/last.pt")
 
if __name__ == '__main__':
   # Training.
   results = model.train(
      data='D:/COS_YOLOv8/240529_COS_bbox/data/240529_data.yaml',
      imgsz=640,
      epochs=1000,
      batch=32,
      name='240529_yolov8l-bbox_custom',
      resume = True
   )
   results = model.val()  # evaluate model performance on the validation set
   print('YOLOv8l-bbox 640px training 완료')
   
#    results = model.train(
#       data='240115_custom_data.yaml',
#       single_cls=True,
#       imgsz=640,
#       epochs=100,
#       batch=32,
#       name='240115_yolov8n_640_custom'
#    )
#    results = model.val()  # evaluate model performance on the validation set
#    print('YOLOv8n 640px training 완료')