# YOLOv5_HNet

EN ｜ [中文](README_CH.md)

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

## Solution3: Network Slim 轻量化网络设计

**思路：**

想法来源于[[1807.11164] ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design](https://arxiv.org/abs/1807.11164) 这篇论文。由于之前的测试都集中在对于准确率的提升上，以上提升准确率方法带来的弊端就是网络结构成倍增加，由此就需要一种能够轻量化网络的思路，我的想法是是否能够基于以上的有效的提升思路，再结合轻量化网络的方法，实现一种速度和准确率的balance。

**改进要点：**
- G1. 卷积层的输入特征channel和输出特征channel要尽量相等；  
- G2. 尽量不要使用组卷积，或者组卷积g尽量小；  
- G3. 网络分支要尽量少，避免并行结构；  
- G4. Element-Wise的操作要尽量少，如：ReLU、ADD、逐点卷积等；

### shuffleNetV2

1. 替换backbone和后面所有C3结构为Shuffle Block，尽量减少非必要卷积操作。
2. 尽量减少如SPP此类的并行操作结构
3. head中的C3改为DWConv结构
4. 两级结构尝试：stage1为仅处理backbone部分，stage2为同时处理backbone和neck部分

### MobileNetV3

1. 使用MobileNetV3替换主要的特征提取backbone

**训练结果：**

| type        | model                | parameters | FLOPs | pt_size | mAP@0.5 | mAP@0.5~0.95 | FPS  |
| ----------- | -------------------- | ---------- | ----- | ------- | ------- | ------------ | ---- |
| Power Plant | YOLOv5s_baseline     | 7.06M      | 16.5G | 14.3MB  | 0.75    | 0.484        | 30.5 |
| Power Plant | ShuffuleNetV2        | 0.7M       | 2.6G  | 1.6MB   | 0.586   | 0.299        | 37   |
| Power Plant | bifpn-cbam-tph-slim  | 45M        | 92.8G | 72.5MB  | 0.758   | 0.415        | 11   |
| Car Factory | YOLOv5s_baseline     | 7.06M      | 16.5G | 14.4MB  | 0.83    | 0.585        | 30.5 |
| Car Factory | ShuffuleNetV2_stage1 | 4.8M       | 16.5G | 8.3MB   | 0.796   | 0.515        | 25   |
| Car Factory | ShuffuleNetV2_stage2 | 0.7M       | 16.5G | 1.7MB   | 0.768   | 0.523        | 38   |
| Car Factory | bifpn-cbam-tph-slim  | 45M        | 92.8G | 72.5MB  | 0.86    | 0.57         | 12   |
| Car Factory | mobileNetV3          | 5.6M       | 12.8G | 3MB     | 0.47    | 0.229         | 35   |

**分析：**

的确经过对应的设计后，在保证检测精度没有大幅下降的前提下，减少了一半多的算力需要，提升了一倍的FPS（但也就只有10帧...）。理论上所需的计算量与baseline相比还是差了很多，效果有提升但也比较小，从此我便开始思考以上这些修改方式造成这种结果的原因。

后来再次做了一个纯ShuffleNetV2 backbone的实验后，发现针对安全帽这种特点的数据，原本的网络尤其是骨干特征提取网络backbone部分是非常优秀的，能够平衡特征提取的精度和速度，因此不应该针对这一块进行大改；同时，注意力机制这种结构对于目前的安全帽检测算法来说，增加算力，减少速度的影响要大于准确率提升带来的优势，所以要根据需求进行tradeoff，目前使用这些没有一个兼顾速度和准确的方法。

## Solution4: Conv卷积核 + FPN修改

**思路：**

从以上的三个solutions可以基本总结出一个情况，针对安全帽这类物体，YOLOv5原始的backbone效果已经足够好，有了足够的balance，因此结构不能进行大改；同时，在此基础上提高准确率的方法就导向了两种方法：一是改进卷积核，二是针对Neck中的FPN进行修改，并在此基础上做准确率和速度的balance

**改进要点：**

base modification：

1. 将一部分backbone中经过C3卷积核之前，添加新的Focus-Conv卷积核，替代全部之前的stride卷积层和池化层，用于进一步在像素中提取信息，提供给下一层卷积操作。常见的stride卷积操作会导致细粒度信息的丢失和学习效率较低的特征表示，从这一点出发，为了在使用类似stride的同时避免对小物体/低分辨率情况下的特征丢失。
2. 在Neck部分concat之前添加一层Focus-Conv卷积，用以增加特征提取，同时减弱stride操作带来的副作用。

advanced modification：

3. 在base基础上将FPN更换为Bi-FPN结构，实现更多层次的特征提取。

**训练结果：**

| type        | model               | parameters | FLOPs | pt_size | mAP@0.5 | mAP@0.5~0.95 | FPS    |
| ----------- | ------------------- | ---------- | ----- | ------- | ------- | ------------ | --- |
| Power Plant | YOLOv5s_baseline    | 7.06M      | 16.5G | 14.3MB  | 0.75    | 0.484        |  30.5   |
| Power Plant | focus-conv-yolov5 | 8.56M        | 28.8G | 17.3MB  | 0.806   | 0.485        | 31    |
| Car Factory | YOLOv5s_baseline    | 7.06M      | 16.5G | 14.4MB  | 0.83    | 0.585        |  30.5   |
| Car Factory | focus-conv-yolov5 | 8.56M        | 28.8G | 17.3MB  | 0.875    | 0.573         | 31    |

**分析：**

在没有明显增加参数量及FLOPs的情况下，发现这种基于focus结构改进的focus-conv卷积核能够在针对安全帽检测的应用场景中有比较好的表现，且能够平衡准确率和检测速度，是一个比较可行的优化方案。


# Reference
[GitHub - ultralytics/yolov5: YOLOv5 🚀 in PyTorch > ONNX > CoreML > TFLite](https://github.com/ultralytics/yolov5)
[GitHub - HuKai97/yolov5-5.x-annotations: 一个基于yolov5-5.0的中文注释版本！](https://github.com/HuKai97/yolov5-5.x-annotations)
[GitHub - ppogg/YOLOv5-Lite: 🍅🍅🍅YOLOv5-Lite: lighter, faster and easier to deploy. Evolved from yolov5 and the size of model is only 930+kb (int8) and 1.7M (fp16). It can reach 10+ FPS on the Raspberry Pi 4B when the input size is 320×320~](https://github.com/ppogg/YOLOv5-Lite)
[GitHub - HuKai97/YOLOv5-ShuffleNetv2: YOLOv5的轻量化改进(蜂巢检测项目)](https://github.com/HuKai97/YOLOv5-ShuffleNetv2)