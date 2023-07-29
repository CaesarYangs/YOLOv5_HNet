# YOLOv5_HNet

EN ｜ [中文](README.md)

A modified version of YOLOv5 for improved detection of key objects in a private helmet dataset.

The main objective of this project is to enhance the detection performance of safety helmets on a private dataset by combining existing research papers with personal insights and modifications.

The criteria for performance improvement are defined as follows:

- Accuracy
- Speed (detection speed)
- Model size (computational requirements)

The dataset used in this project is highly diverse and includes various scenarios for helmet detection. Extensive experiments and necessary ablation studies have been conducted on these valuable datasets.

Tip: This project is suitable for individuals who have a thorough understanding of YOLOv5 and its subsequent versions. It is recommended to read and understand the overall source code and essential details by referring to the comprehensive annotated version provided by HuKai in the GitHub repository [GitHub - HuKai97/yolov5-5.x-annotations: 一个基于yolov5-5.0的中文注释版本！](https://github.com/HuKai97/yolov5-5.x-annotations). I am grateful for this project as it has allowed me to quickly comprehend and make modifications in the early stages.

## Solution 1: Changing Anchors and Adding Small Object Detection Layer

**Approach:**

After analyzing the dataset, it was observed that most of the images captured by surveillance cameras show objects at a considerable distance, resulting in smaller objects. The first direction to focus on is optimizing the detection of these small objects.

**Improvements:**

1. Adding a new anchor specifically for small objects.
   
```yaml
[5, 6, 8, 14, 15, 11] #new_anchor
```

2. Modifying the Feature Pyramid Network (FPN) structure by adding additional layers.
   
   On top of the existing 5-layer FPN, two additional layers are added for dimension 512 and dimension 256 feature extraction. The last layer, which goes through upsampling and feature extraction, is then connected to the detection head, replacing the previous single-layer extraction/upsampling process.

3. Utilizing k-means clustering to reassign anchors based on the specific dataset.
   
   This essentially involves manually verifying the anchor reassignment algorithm, which is a subsequent step after the anchor regression prediction of the YOLOv5 model. This measures the effectiveness of the method for helmet detection in this particular domain.

**Training Results:**

| Type        | Model                     | Parameters | FLOPs  | .pt size | mAP@0.5 | mAP@0.5~0.95 | FPS  |
| ----------- | ------------------------ | ---------- | ------ | -------- | ------- | ------------ | ---- |
| Power Plant | YOLOv5s_baseline         | 7.06M      | 16.5G  | 14.3MB   | 0.75    | 0.484        | 30.5 |
| Power Plant | little_obj_anchor_fpn    | 50.9M      | 192.5G | 102.3MB  | 0.748   | 0.45         | 4.3  |
| Power Plant | little_obj_anchor_kmeans | 7.06M      | 16.5G  | 14.3MB   | 0.75    | 0.473        | 30.5 |
| Power Plant | little_obj_anchor        | 7.06M      | 16.5G  | 14.3MB   | 0.7552  | 0.4867       | 30.5 |

**Analysis:**

This method can be considered as "quantity over quality." Simply stacking additional layers for feature extraction and detection heads does not yield significant improvements in the specific domain of safety helmet detection. Moreover, it increases the computational requirements nearly tenfold, while the overall accuracy barely improves. It is possible that fine-tuning specific to achieve "Large Model" object detection performance is needed, but blindly stacking layers is not the solution. This serves as a negative example worth contemplating.

However, it does prove that reassigning anchors can be helpful in achieving improvements without adding computational load or modifying the network structure. It demonstrates the usability of remapping anchors for target objects specific to safety helmets in this domain.

## Solution 2: Modifying Neck&Head Structure, Adding Attention Mechanisms, and Transformer

**Approach:**

This method is inspired by the paper [TPH-YOLOv5](https://arxiv.org/abs/2108.11539), which introduces a Transformer Prediction Head to YOLOv5, resulting in improved detection performance. Based on this detection head, I choose to focus on optimizing the FPN structure and incorporating TPH with attention mechanisms, considering the characteristics of the dataset at hand.

**Improvements:**

1. Modifying the FPN structure to Bi-FPH (Weighted Bidirectional Feature Pyramid Network) to enhance the accuracy and precision of feature fusion, and to some extent, address the issue of small objects.
2. Adding CBAM (Convolutional Block Attention Module), a simple and effective attention mechanism for feed-forward convolutional neural networks. CBAM is a lightweight and versatile module that can be seamlessly integrated into any CNN architecture, and can be trained end-to-end with the base CNN.
3. Finally, combining the TPH detection head to form a four-headed detection head.

**Training Results:**

| Type        | Model            | Parameters | FLOPs  | .pt size | mAP@0.5 | mAP@0.5~0.95 | FPS   |
| ----------- | ---------------- | ---------- | ------ | -------- | ------- | ------------ | ----- |
| Power Plant | YOLOv5s_baseline | 7.06M      | 16.5G  | 14.3MB   | 0.75    | 0.484        | 30.5  |
| Power Plant | bifpn-cbam-tph   | 45M        | 180G   | 87.5MB   | 0.76    | 0.388        | 10.0  |
| Car Factory | YOLOv5s_baseline | 7.06M      | 16.5G  | 14.4MB   | 0.83    | 0.585        | 30.0  |
| Car Factory | bifpn-cbam-tph   | 45M        | 180G   | 87.5MB   | 0.85    | 0.574        | 12.0  |

**Analysis:**

By incorporating a series of mechanisms, there is indeed an improvement in overall accuracy. However, this comes at the cost of significantly increased computational requirements and slower convergence speed. Additionally, the detection speed is greatly reduced. I have some doubts about this, as it seems that excessive feature extraction or attention mechanisms do not provide significant improvements or significance in the detection of safety helmets and similar objects.

## Solution 3: Network Slim - Lightweight Network Design

**Approach:**

The idea behind this solution comes from the paper [ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design](https://arxiv.org/abs/1807.11164). While previous experiments focused on improving accuracy, the drawback of these methods is that they significantly increase the network's structure. Therefore, there is a need for a lightweight network design that can balance speed and accuracy using the effective improvement strategies mentioned earlier.

**Key Improvements:**
- G1. The input and output feature channels of convolutional layers should be approximately equal.
- G2. Avoid using group convolutions or use a small number of groups.
- G3. Aim to minimize the number of network branches to avoid parallel structures.
- G4. Minimize the use of element-wise operations such as ReLU, ADD, and pointwise convolutions.

### ShufflenetV2

1. Replace the backbone and all C3 structures with Shuffle Blocks to minimize non-essential convolutional operations.
2. Minimize the use of parallel operations like SPP.
3. Replace the C3 in the head with DWConv structure.
4. Try a two-stage approach: stage 1 only processes the backbone, while stage 2 simultaneously processes the backbone and neck.

### MobileNetV3

1. Use MobileNetV3 as the main backbone for feature extraction.

**Training Results:**

| Type        | Model                | Parameters | FLOPs | .pt size | mAP@0.5 | mAP@0.5~0.95 | FPS  |
| ----------- | -------------------- | ---------- | ----- | -------- | ------- | ------------ | ---- |
| Power Plant | YOLOv5s_baseline     | 7.06M      | 16.5G | 14.3MB   | 0.75    | 0.484        | 30.5 |
| Power Plant | ShuffuleNetV2        | 0.7M       | 2.6G  | 1.6MB    | 0.586   | 0.299        | 37   |
| Power Plant | bifpn-cbam-tph-slim  | 45M        | 92.8G | 72.5MB   | 0.758   | 0.415        | 11   |
| Car Factory | YOLOv5s_baseline     | 7.06M      | 16.5G | 14.4MB   | 0.83    | 0.585        | 30.5 |
| Car Factory | ShuffuleNetV2_stage1 | 4.8M       | 16.5G | 8.3MB    | 0.796   | 0.515        | 25   |
| Car Factory | ShuffuleNetV2_stage2 | 0.7M       | 16.5G | 1.7MB    | 0.768   | 0.523        | 38   |
| Car Factory | bifpn-cbam-tph-slim  | 45M        | 92.8G | 72.5MB   | 0.86    | 0.57         | 12   |
| Car Factory | MobileNetV3          | 5.6M       | 12.8G | 3MB      | 0.47    | 0.229        | 35   |

**Analysis:**

After implementing the designed modifications, the accuracy remained similar without a significant drop, while the computational requirements were reduced by more than half, resulting in twice the FPS (although still only 10 frames per second...). In theory, the computational requirement is much lower compared to the baseline, but the improvement is relatively small. This made me question the reasons behind these results.

I conducted another experiment using only ShuffleNetV2 as the backbone and found that the original network, especially the backbone for feature extraction, is already well balanced in terms of accuracy and speed for safety helmet detection. Therefore, major changes should not be made to this part. Additionally, attention mechanisms tend to have a negative impact on speed and cause increased computational requirements, which outweigh the benefits of accuracy improvements in the case of safety helmet detection. Thus, it is necessary to trade off between speed and accuracy based on specific requirements. Currently, there is no single method that balances both speed and accuracy effectively.

## Solution 4: Convolutional Kernel + FPN Modification

**Approach:**

From the previous three solutions, we can conclude that the YOLOv5 backbone is already well balanced for safety helmet detection. Therefore, we should avoid making major changes to the structure. The methods that improve accuracy can be categorized into two approaches: (1) modifying the convolutional kernels, and (2) modifying the FPN in the Neck, while maintaining a balance between accuracy and speed.

**Key Improvements:**

Base Modification:

1. Add a new Focus-Conv layer before some of the C3 convolutional layers in the backbone to extract pixel-level information and provide it to the next layer. This is done to avoid feature loss and improve the feature representation efficiency caused by stride convolutions, which can result in the loss of fine-grained information and low learning efficiency in low-resolution cases.
2. Add a Focus-Conv layer before the concatenation in the Neck to increase feature extraction and mitigate the side effects of stride operations.

Advanced Modification:

3. Replace FPN with the Bi-FPN structure to enable multi-level feature extraction.

**Training Results:**

| Type        | Model              | Parameters | FLOPs | .pt Size | mAP@0.5 | mAP@0.5~0.95 | FPS |
| ----------- | ------------------ | ---------- | ----- | -------- | ------- | ------------ | --- |
| Power Plant | YOLOv5s_baseline   | 7.06M      | 16.5G | 14.3MB   | 0.75    | 0.484        | 30.5 |
| Power Plant | focus-conv-yolov5 | 8.56M      | 28.8G | 17.3MB   | 0.806   | 0.485        | 31   |
| Car Factory | YOLOv5s_baseline   | 7.06M      | 16.5G | 14.4MB   | 0.83    | 0.585        | 30.5 |
| Car Factory | focus-conv-yolov5 | 8.56M      | 28.8G | 17.3MB   | 0.875   | 0.573        | 31   |

**Analysis:**

By introducing the modified focus-convolutional kernel without significantly increasing the number of parameters and FLOPs, we achieved good performance for safety helmet detection. This modification maintains a balance between accuracy and detection speed, making it a feasible optimization solution.


# Reference
[GitHub - ultralytics/yolov5: YOLOv5 🚀 in PyTorch > ONNX > CoreML > TFLite](https://github.com/ultralytics/yolov5)
[GitHub - HuKai97/yolov5-5.x-annotations: 一个基于yolov5-5.0的中文注释版本！](https://github.com/HuKai97/yolov5-5.x-annotations)
[GitHub - ppogg/YOLOv5-Lite: 🍅🍅🍅YOLOv5-Lite: lighter, faster and easier to deploy. Evolved from yolov5 and the size of model is only 930+kb (int8) and 1.7M (fp16). It can reach 10+ FPS on the Raspberry Pi 4B when the input size is 320×320~](https://github.com/ppogg/YOLOv5-Lite)
[GitHub - HuKai97/YOLOv5-ShuffleNetv2: YOLOv5的轻量化改进(蜂巢检测项目)](https://github.com/HuKai97/YOLOv5-ShuffleNetv2)