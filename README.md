1. Introduction

This program implements the method described in the paper "Enhanced Small Apple Detection on Trees Using MODA-YOLO: An Optimized YOLOv5n Model", submitted to The Visual Computer journal.

The OHA-YOLO model is designed for automated apple detection in orchards, aiming to improve fruit recognition accuracy and efficiency. The model incorporates multiple optimization strategies to enhance small object detection, particularly for densely packed apples on trees.

2. Key Algorithmic Enhancements

To achieve high-precision apple detection, the model integrates several advanced techniques:

  1）Lightweight Backbone – MobileNetV2:Reduces computational cost while maintaining feature extraction efficiency by utilizing depthwise separable convolutions, which replace traditional convolutions with a factorized version, significantly decreasing the number of parameters and operations.Enhances real-time detection capability by leveraging an inverted residual structure with linear bottlenecks, allowing for efficient feature reuse and reducing information loss.Improves small object detection by preserving spatial details through its narrower bottleneck layers, making it suitable for detecting densely packed apples in orchard environments.

  2）Omni-Dimensional Dynamic Convolution (ODConv):Introduces learnable deformation modules.Improves adaptability to varying object shapes and sizes.Enhances feature representation, particularly for small apples. Dynamic Head (DYHead) with Multi-Head Self-Attention:Strengthens feature fusion across scales.Improves the ability to capture detailed features of small apples.
 
  3）Optimal Transport Assignment (OTA) for Label Matching: Optimizes target assignment in dense object scenarios. Ensures better alignment between predictions and ground truth. Through these enhancements, the model achieves 94.4% mAP, surpassing the original baseline by 7.7%.

3. Program Installation & Execution

Before running the program, ensure the required dependencies are installed:  pip install -r requirements.txt

  1）Training the Model Open train.py. Configure necessary paths: Dataset: /data/appletree.yaml Define dataset paths: train: /path/to/train/images val: /path/to/validation/images test: /path/to/test/images Specify class names:nc: 1  # Number of classes names: ['apple']  Model file: /models/yolov5-custom.yaml  Hyperparameters: /data/hyps/hyp.scratch-low.yaml  Adjust training parameters as needed: epochs=300 (Number of training iterations) batch_size=16 (Number of images processed at a time) img_size=640 (Input image resolution) Run the training script: python train.py --data /data/appletree.yaml --cfg /models/yolov5-custom.yaml --hyp /data/hyps/hyp.scratch-low.yaml --epochs 300 --batch-size 16 --img 640Training results, including model weights, loss curves, and logs, will be stored in runs/train/. The best model checkpoint will be saved as runs/train/exp/weights/best.pt. If training is interrupted, resume training by specifying the last saved checkpoint: python train.py --resume runs/train/exp/weights/last.pt
  
  2） Model Evaluation Run val.py to evaluate the trained model: python val.py --weights runs/train/exp/weights/best.pt --data /data/appletree.yaml --img 640Key metrics, including mAP, Precision, and Recall, will be displayed in the console.Detailed evaluation results, including confusion matrices and per-class performance, will be saved in runs/val/.To visualize precision-recall curves and loss evolution, use TensorBoard:tensorboard --logdir=runs/train/
  
  3） Object Detection Run detect.py for real-time or batch detection: python detect.py --weights runs/train/exp/weights/best.pt --source path_to_image_or_video --img 640 --conf 0.4 --iou 0.5Supported input sources: Single image: --source image.jpg Video file: --source video.mp4 Webcam: --source 0 Adjust confidence and IoU thresholds as needed (--conf and --iou). Detection results, including bounding boxes and class labels, will be saved in runs/detect/exp/. The detected images/videos will be stored with annotated bounding boxes for further analysis.

4.  Analysis

 The following figure shows the change curves of various metrics during the model training process, including Box Loss, Objectness Loss, Classification Loss during both the training and validation phases, as well as Precision, Recall, and mean Average Precision (mAP).
 
 <img width="454" alt="image" src="https://github.com/user-attachments/assets/6ad9c1e2-3ec8-469f-920a-0e0f420a18d6" />

 Curve of each index with the increasing of training rounds during the model training

 The figure shows the change curves of key evaluation metrics during the training process, including F1-score vs. confidence, Precision vs. confidence, Recall vs. confidence, and Precision-Recall (PR curve). During the detection process of the dense small target apple dataset using the OHA-YOLO model, all metrics exhibited favorable trends. The F1-score vs. confidence curve shows that, with the adjustment of the confidence threshold, the F1-score reaches a peak within a certain range, indicating that the model performs excellently in balancing precision and recall.

 <img width="426" alt="image" src="https://github.com/user-attachments/assets/83e6f09f-d714-4a73-ad4f-d7d2f104471a" />

Model training evaluation metrics (a) F1 value-confidence (b) precision-confidence (c) recall-confidence (d) precision-recall

In summary, the OHA-YOLO model provides an accurate and efficient solution for the detection of apples in orchards, improving picking precision and efficiency, reducing human intervention, and promoting the development of agricultural automation and intelligence.
