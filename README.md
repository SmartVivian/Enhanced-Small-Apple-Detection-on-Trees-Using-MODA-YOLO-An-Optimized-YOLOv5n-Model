  This program implements the method described in the paper "Enhanced Small Apple Detection on Trees Using MODA-YOLO: An Optimized YOLOv5n Model", published in The Visual Computer. 
 
  The model aims to achieve automated apple picking in orchards and accurately locate apples on apple trees. To achieve this, the model first adopts the lightweight backbone network MobileNetv2 and incorporates the Omni-Dimensional Dynamic Convolution (ODConv) algorithm. This approach enhances small object detection capabilities through a learnable deformation module. Additionally, the model integrates the Dynamic Head (DYHead) with a multi-head self-attention mechanism to improve detection accuracy. Finally, the Optimal Transport Assignment (OTA) algorithm is used to optimize label matching in dense target scenarios, significantly enhancing the detection performance of densely packed fruits. The model achieves an mAP of 94.4%, a 7.7% improvement over the original model.

Program Operation Steps:
   After opening the program, install the required libraries by running   pip install -r requirements.txt.
     
  1)Train: Once the installation is complete, open train.py, configure the dataset path /data/appletree.yaml, the model file path /models/yolov5-custom.yaml, and the hyperparameters path /data/hyps/hyp.scratch-low.yaml. You can set the number of training epochs, batch size, and other parameters as needed, and save the training results in the runs/train directory.
  
  2)Test: Run val.py, select the trained model, and set the dataset to be tested. Choose the test task to evaluate the model on the test set. The results will be saved in the runs folder.
  
  3)Detect: Run detect.py, and select an image, video, or real-time camera feed for detection. The detection results will be displayed on the image, video, or camera interface, allowing users to view them conveniently.

  The following figure shows the change curves of various metrics during the model training process, including Box Loss, Objectness Loss, Classification Loss during both the training and validation phases, as well as Precision, Recall, and mean Average Precision (mAP).

<img width="454" alt="image" src="https://github.com/user-attachments/assets/4d3360d0-185d-4e68-88ea-c4c2be34143d" />

Curve of each index with the increasing of training rounds during the model training

  The figure shows the change curves of key evaluation metrics during the training process, including F1-score vs. confidence, Precision vs. confidence, Recall vs. confidence, and Precision-Recall (PR curve). During the detection process of the dense small target apple dataset using the OHA-YOLO model, all metrics exhibited favorable trends. The F1-score vs. confidence curve shows that, with the adjustment of the confidence threshold, the F1-score reaches a peak within a certain range, indicating that the model performs excellently in balancing precision and recall.

<img width="454" alt="image" src="https://github.com/user-attachments/assets/20ddba83-77d0-47b3-9c7f-cfe63eab12d8" />

Model training evaluation metrics (a) F1 value-confidence (b) precision-confidence (c) recall-confidence (d) precision-recall

  In summary, the OHA-YOLO model provides an accurate and efficient solution for the detection of apples in orchards, improving picking precision and efficiency, reducing human intervention, and promoting the development of agricultural automation and intelligence.

