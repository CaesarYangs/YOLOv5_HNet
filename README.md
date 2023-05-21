# YOLOv5_HNet
A YOLOv5 modification net for key objects on private helmet dataset

本项目的主要目标是利用一些已有的论文方案，结合我自己的一些思考和改动，实现对于私人数据集上的安全帽检测的效果提升。

效果提升的标准定义：

- 准确率
- 速度（检测速度）
- 模型大小（所需算力）

本项目所基于的数据集种类、数量都极其复杂，涉及多场景的情况处理。也在这些珍贵的数据集上进行了大量的实验及必须的消融实验。

tips：本项目适合完全了解YOLOv5及以上版本源码的同学阅读及使用。建议可以先通过[GitHub - HuKai97/yolov5-5.x-annotations: 一个基于yolov5-5.0的中文注释版本！](https://github.com/HuKai97/yolov5-5.x-annotations) HuKai大神之前做过的这个非常详尽的标注版本进行整体源码和必要细节阅读理解。也非常感谢这个项目让我在早期能够快速吃透并上手修改工作。

## Solution1: 改变Anchor，添加小目标检测层

**思路：**

根据对数据集的分析，发现更多的情况是监控拍摄到的图片，结果就是目标的距离很远，物体较小。自然想到的第一个方向就是针对这些小目标着重优化。

**改进要点：**

1. 添加一层新的小目标Anchor。
	
```yaml
[5, 6, 8, 14, 15, 11] #new_anchor
```

2. 修改FPN结构，增加层数。
	
	在原来5层FPN的基础上增加两层：分别是对dim=512及dim=256维度特征的再次提取，经过再次上采样提取过的最后一层连接到检测头，替换原来的单层提取/上采。

3. 利用kmeans，针对特定数据集情况做anchor重新聚类
	
	本质上是针对后期YOLOv5模型自带的anchor回归预测的一个手动验证，看经过我解耦后的单独anchor重聚类算法能否对基础的YOLO模型有所提升，以验证这个方法对于安全帽检测这个特定领域的有效性。


**训练结果：**

| type        | model                    | parameters | FLOPs  | pt_size | mAP@0.5 | mAP@0.5~0.95 | FPS  |
| ----------- | ------------------------ | ---------- | ------ | ------- | ------- | ------------ | ---- |
| Power Plant | YOLOv5s_baseline         | 7.06M      | 16.5G  | 14.3MB  | 0.75    | 0.484        | 30.5 |
| Power Plant | little_obj_anchor_fpn    | 50.9M      | 192.5G | 102.3MB | 0.748   | 0.45         | 4.3  |
| Power Plant | little_obj_anchor_kmeans | 7.06M      | 16.5G  | 14.3MB  | 0.75  | 0.473      | 30.5 |
| Power Plant | little_obj_anchor        | 7.06M      | 16.5G  | 14.3MB  | 0.7552  | 0.4867       | 30.5 |


**分析：**

此方法可谓”大而无当“。仅仅通过堆砌特征层，堆砌检测头的力大砖飞式改进并不能在安全帽这种特定领域下有很好的效果，而且徒增了接近十倍的算力需求，反而整体的准确率几乎没有提升。也有可能是需要特定的fine-tunning来实现目标检测上的“Large Model”，但肯定不会是无脑堆砌。作为一个反例值得思考。

同时，证明了anchors重聚类对于提升还是比较有帮助的，不增加任何计算压力，不修改任何网络结构的情况下也能实现一些提升。证明了remapping anchors对于安全帽这种特定领域下的目标的可用性。

## Solution2: 改变Neck&Head结构，加入注意力机制，Transformer等

**思路：**

此方法的灵感来源于[TPH-YOLOv5](https://arxiv.org/abs/2108.11539) 这篇论文，原作者给YOLOv5加入了Transformer Prediction Head，最后的检测有了比较好的提高。我便基于这个检测头，根据我手中数据集的特点，选择重点优化FPN结构，并结合TPH和一些注意力机制。

**改进要点：**

1. 改变FPN结构为Bi-FPH（加权双向特征金字塔网络），以期提升特征融合的精度及准确度，并一定程度上解决小目标问题。
2. 增加CBAM（卷积注意力模块），这是一种用于前馈卷积神经网络的简单而有效的注意力模块。CBAM是轻量级的通用模块，因此可以忽略的该模块的开销而将其无缝集成到任何CNN架构中，并且可以与基础CNN一起进行端到端训练。
3. 最后结合TPH检测头，最后组合为四头检测Head。

**训练结果：**

| type        | model            | parameters | FLOPs | pt_size | mAP@0.5 | mAP@0.5~0.95 | FPS    |
| ----------- | ---------------- | ---------- | ----- | ------- | ------- | ------------ | --- |
| Power Plant | YOLOv5s_baseline | 7.06M      | 16.5G | 14.3MB  | 0.75    | 0.484        |    30.5 |
| Power Plant | bifpn-cbam-tph   | 45M        | 180G  | 87.5MB  | 0.76    | 0.388        | 10    |
| Car Factory | YOLOv5s_baseline | 7.06M      | 16.5G | 14.4MB  | 0.83    | 0.585        |  30   |
| Car Factory | bifpn-cbam-tph   | 45M        | 180G  | 87.5MB  | 0.85    | 0.574       | 12    |

**分析：**

加入了一系列机制后，整体的准确率的确有了提升，但是相对的就是算力的大幅提高以及收敛速度显著下降。同时检测速度也极大下降。我也对此有一些疑问，貌似过分的特征提取or注意力机制对于安全帽这类物体检测来说提升效果和意义并不明显。

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
