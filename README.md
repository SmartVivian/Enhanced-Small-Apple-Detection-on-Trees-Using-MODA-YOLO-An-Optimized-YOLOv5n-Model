1. Introduction
   
This program implements the method described in the paper "Enhanced Small Apple Detection on Trees Using MODA-YOLO: An Optimized YOLOv5n Model", submitted to The Visual Computer journal.

The OHA-YOLO model is designed for automated apple detection in orchards, aiming to improve fruit recognition accuracy and efficiency. The model incorporates multiple optimization strategies to enhance small object detection, particularly for densely packed apples on trees.

2. Key Algorithmic Enhancements

To achieve high-precision apple detection, the model integrates several advanced techniques:

1)Lightweight Backbone – MobileNetV2

MobileNetV2 is a highly efficient lightweight neural network architecture designed for mobile and embedded devices. It aims to reduce computational complexity while retaining strong feature extraction capability.

(1) Depthwise Separable Convolutions

MobileNetV2 uses depthwise separable convolutions, which break down the traditional convolution operation into two steps:

•Depthwise Convolution: A separate convolution is applied to each input channel.
•Pointwise Convolution: A standard 1×1 convolution is used to combine the output channels.

This factorization drastically reduces the number of parameters and operations compared to regular convolutions, making the network computationally efficient.

(2) Inverted Residual Structure with Linear Bottlenecks

The core of MobileNetV2 is the inverted residual structure, which uses a linear bottleneck to optimize the flow of information:

•The input feature map undergoes a depthwise separable convolution in the high-dimensional space, followed by a linear bottleneck in the lower-dimensional space.
•This structure minimizes information loss while reducing computational cost and makes the network more efficient.

The design ensures better feature reuse and reduces information loss, which is especially useful for real-time detection on mobile devices.

(3)Improvement for Small Object Detection

MobileNetV2 has narrower bottleneck layers, which help to preserve spatial details in smaller objects. This characteristic is crucial for detecting small, densely packed objects like apples on trees, where fine spatial details must be maintained for accurate detection.

2)Omni-Dimensional Dynamic Convolution (ODConv)
ODConv is an innovative convolutional mechanism designed to improve adaptability to varying object shapes and sizes, making it especially useful for detecting small objects.
(1)Learnable Deformation Modules

ODConv introduces learnable deformation modules that allow convolution kernels to dynamically adjust according to the shape of the object. During training, the network learns optimal kernel shapes tailored to specific object characteristics, improving the adaptability of the model.
This makes ODConv highly flexible, enabling the network to adjust to different object shapes in real-time.
(2)Improved Adaptability
Unlike traditional convolution, where kernels are fixed, ODConv can adapt to a range of object shapes and sizes. This flexibility is particularly advantageous for tasks like detecting small apples in dense orchard environments, where traditional methods might struggle.
(3)Enhanced Feature Representation
ODConv enhances feature extraction, particularly for small objects, by enabling the network to focus on the most relevant parts of an object. The dynamic nature of the convolution kernel allows the model to capture finer details of small apples that would otherwise be missed in traditional convolution operations.
  
3)Dynamic Head (DYHead) with Multi-Head Self-Attention

DYHead is a novel detection head that incorporates multi-head self-attention mechanisms to improve the fusion of features across different scales, particularly for small object detection.

(1) Feature Fusion Across Scales

DYHead strengthens the fusion of features from multiple scales by using multi-head self-attention. In traditional detection heads, features from different layers may not be fully leveraged, but the attention mechanism allows DYHead to focus on important features at different scales.

This makes it particularly effective for detecting small objects like apples, which may appear in various sizes depending on the camera's perspective.

(2) Improved Ability to Capture Detailed Features

The self-attention mechanism in DYHead enables the network to better capture detailed features of small objects. By dynamically adjusting the attention to different parts of the input, DYHead can focus on the most critical aspects of the image, making it more efficient at detecting small and densely packed objects in the scene.

4)Optimal Transport Assignment (OTA) for Label Matching

Optimal Transport Assignment (OTA) is a label matching strategy that optimizes the assignment of ground truth boxes to predicted boxes, particularly in dense object detection scenarios.

(1)Label Matching Optimization

In traditional object detection models, IoU (Intersection over Union) is used to match predictions with ground truth. However, in dense object scenarios (like detecting apples on a tree), many targets may have low IoU values, making it difficult to match predictions accurately. OTA optimizes label matching by leveraging optimal transport theory, which ensures a more accurate and stable assignment of predictions to ground truth.

(2)Improved Alignment Between Predictions and Ground Truth

OTA improves the alignment between predicted bounding boxes and ground truth in dense object scenarios. By solving the optimal transport problem, it minimizes the cost of mismatching labels, leading to more accurate predictions, especially in complex environments where objects are densely packed.

3. Program Installation & Execution

Before running the program, ensure the required dependencies are installed:  pip install -r requirements.txt

  1）Training the Model： Open train.py. Configure necessary paths: Dataset: /data/appletree.yaml Define dataset paths: train: /path/to/train/images val: /path/to/validation/images test: /path/to/test/images Specify class names:nc: 1  # Number of classes names: ['apple']  Model file: /models/yolov5-custom.yaml  Hyperparameters: /data/hyps/hyp.scratch-low.yaml  Adjust training parameters as needed: epochs=300 (Number of training iterations) batch_size=16 (Number of images processed at a time) img_size=640 (Input image resolution) Run the training script: python train.py --data /data/appletree.yaml --cfg /models/yolov5-custom.yaml --hyp /data/hyps/hyp.scratch-low.yaml --epochs 300 --batch-size 16 --img 640Training results, including model weights, loss curves, and logs, will be stored in runs/train/. The best model checkpoint will be saved as runs/train/exp/weights/best.pt. If training is interrupted, resume training by specifying the last saved checkpoint: python train.py --resume runs/train/exp/weights/last.pt
  
  2） Model Evaluation： Run val.py to evaluate the trained model: python val.py --weights runs/train/exp/weights/best.pt --data /data/appletree.yaml --img 640 Key metrics, including mAP, Precision, and Recall, will be displayed in the console.Detailed evaluation results, including confusion matrices and per-class performance, will be saved in runs/val/.To visualize precision-recall curves and loss evolution, use TensorBoard:tensorboard --logdir=runs/train/
  
  3） Object Detection： Run detect.py for real-time or batch detection: python detect.py --weights runs/train/exp/weights/best.pt --source path_to_image_or_video --img 640 --conf 0.4 --iou 0.5 Supported input sources: Single image: --source image.jpg Video file: --source video.mp4 Webcam: --source 0 Adjust confidence and IoU thresholds as needed (--conf and --iou). Detection results, including bounding boxes and class labels, will be saved in runs/detect/exp/. The detected images/videos will be stored with annotated bounding boxes for further analysis.

4.  Analysis

 The following figure shows the change curves of various metrics during the model training process, including Box Loss, Objectness Loss, Classification Loss during both the training and validation phases, as well as Precision, Recall, and mean Average Precision (mAP).
 
 <img width="454" alt="image" src="https://github.com/user-attachments/assets/6ad9c1e2-3ec8-469f-920a-0e0f420a18d6" />

 Curve of each index with the increasing of training rounds during the model training

 The figure shows the change curves of key evaluation metrics during the training process, including F1-score vs. confidence, Precision vs. confidence, Recall vs. confidence, and Precision-Recall (PR curve). During the detection process of the dense small target apple dataset using the OHA-YOLO model, all metrics exhibited favorable trends. The F1-score vs. confidence curve shows that, with the adjustment of the confidence threshold, the F1-score reaches a peak within a certain range, indicating that the model performs excellently in balancing precision and recall.

 <img width="426" alt="image" src="https://github.com/user-attachments/assets/83e6f09f-d714-4a73-ad4f-d7d2f104471a" />

Model training evaluation metrics (a) F1 value-confidence (b) precision-confidence (c) recall-confidence (d) precision-recall

In summary, the OHA-YOLO model provides an accurate and efficient solution for the detection of apples in orchards, improving picking precision and efficiency, reducing human intervention, and promoting the development of agricultural automation and intelligence.
