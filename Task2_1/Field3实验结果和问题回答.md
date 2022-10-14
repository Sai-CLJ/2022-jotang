# Field 3

## 1 理论问题

+ **GCN的网络深度越深越好吗？随着深度增加会不会出现什么问题？**

类似于CNN，GCN的性能在网络深度到达一定程度时不增反降。出现的问题就是**过平滑**。不可忽略的是，一定的深度在很多场景中也是必要的，因为刻画图的拓扑结构需要足够深的架构，反之学习器只能学习到“低维特征”。



+ **过平滑是什么？为什么GCN加深会导致过平滑？**

直观地说，过平滑是**不同节点的特征收敛于相同的值**的现象。同一个连通分量内的节点，经过多层GCN后，输出的特征几乎完全相同。

过平滑原因的直观理解：图卷积本质上是在**聚合**邻居节点的信息。网络深度每加深一层，任一节点能聚合的邻居节点的阶数就升高一层，直至整个连通分量内，所有节点的信息都被彼此互相聚合。过度聚合就会导致，原本特征相差很大的节点也被聚合在一起，致使无法有效区分它们。

从理论上讲，图卷积就是一种拉普拉斯平滑，对节点的特征进行低频滤波。这样可以使图信号变得平滑，但是多次平滑后就会过度平滑了。论文中证明，对几种典型的GCN而言，当采用无限多的层数时，模型的输出都将**收敛**到一个长方体。



- **文中提出的DropEdge机制是如何工作的？**

DropEdge是指，在**每次训练**时，**随机剔除**输入图中**一定比例的边**。在其特殊形式中，每条边都以固定概率p独立丢弃，p是一个超参数，由验证集确定。

我个人的理解是，p是在验证集上不断调参后找到的较优值。在论文作者的源代码中，并没有自动调参的代码，其sh文件中的p值需要手动输入。也就是说，p是经验性的，对不同的数据集和不同的模型架构，都有不同的最优p值。



+ **TABLE 2 的实验结果说明了什么？（随深度增加，Original和DropEdge的accuracy变化情况）**

![](Field3实验结果和问题回答.assets/694b8e69c5eaed4a976eb74bbd342e6.jpg)

表2如上。

+ 可以看出，DropEdge基本上使得各个模型应用各个层数时，在各个数据集上的表现均得到提升。
+ 与Original相比，采用了DropEdge后，可能增大模型的最优层数，这同时也增加了最优准确率。
+ GCN, GCN-b, ResGCN均出现了随层数增加，准确率大幅下降的情况（无论DropEdge与否）。而对于后两者JKNet和APPNP，DropEdge对其准确率的增幅相对较小，而且它们的准确率在层数增加时并没有明显下降。一个可能的原因是，层数还不够多。



## 2 探究深度对准确率的影响

### 2.1 **‘y’ , ‘ty’ , ‘ally’ , ‘x’ , ‘tx’ , ‘allx’ , ‘graph’ 是什么？**

+ **ind.cora.x** : 训练集节点特征向量，保存对象为：scipy.sparse.csr.csr_matrix，实际展开后大小为： (140, 1433)
+ **ind.cora.tx** : 测试集节点特征向量，保存对象为：scipy.sparse.csr.csr_matrix，实际展开后大小为： (1000, 1433)
+ **ind.cora.allx** : 包含有标签和无标签的训练节点特征向量，保存对象为：scipy.sparse.csr.csr_matrix，实际展开后大小为：(1708, 1433)，可以理解为除测试集以外的其他节点特征集合，训练集是它的子集
+ **ind.cora.y** : one-hot表示的训练节点的标签，保存对象为：numpy.ndarray，shape为(140, 7)
+ **ind.cora.ty** : one-hot表示的测试节点的标签，保存对象为：numpy.ndarray
+ **ind.cora.ally** : one-hot表示的ind.cora.allx对应的标签，保存对象为：numpy.ndarray
+ **ind.cora.graph** : 保存节点之间边的信息，保存格式为：{ 索引 : [ 索引节点的邻居节点 ] }
+ **ind.cora.test.index** : 保存测试集节点的索引，保存对象为：List，用于后面的归纳学习设置。就是1000个点的标号组成的list



### 2.2 **Original GCN的准确率随深度的变化**

仿照论文的思路，将网络层数分别设置为2, 4, 8, 16, 32, 64进行探究。

下面两张图中，从左上至右下依次为2, 4, 8, 16, 32, 64层网络对应的**损失&准确率曲线图**和**节点特征TSNE降维图**。



![](Field3实验结果和问题回答.assets/c974fa3acf7681dcdc6f27cc63dee1b.jpg)

<center><b>图1 损失&准确率曲线图-Oringinal</b></center>

图中蓝色曲线为验证集上的准确率曲线，红色曲线为训练集上的损失函数曲线







![](Field3实验结果和问题回答.assets/ab731c780de366e1ebfb8e72d81bef3.jpg)

<center><b>图2 节点特征TSNE降维图-Oringinal</b></center>





| 网络层数         | 2     | 4     | 8     | 16    | 32            | 64            |
| ---------------- | ----- | ----- | ----- | ----- | ------------- | ------------- |
| **测试集准确率** | 0.796 | 0.788 | 0.771 | 0.694 | 0.710（峰值） | 0.391（峰值） |

<center><b>表1 测试集准确率-Oringinal</b></center>



结合准确率和特征可视化图表，可以发现**随着深度的增加**，测试集上的**准确率呈现下降的趋势**。同时，特征之间的区别也越来越模糊，这与之前理论分析的结果是一致的。

同时可以看出，对于Oringinal GCN而言，采用32和64层网络时，测试集上的准确率均发生了**突变**现象，即反复的突降和突增。我尝试调整优化器的学习率、学习率损失等超参数，并没有解决这一问题。一个可能的原因是，训练集的样本数目太小了，只有140个节点，导致学习误差增大。（相较于cv任务以万为单位的训练集，这样小的训练集在我自己电脑上的训练速度也挺快的，于是就没有使用colab。）

按照论文所述，所有层数均采用400个epochs进行训练。但是对大层数而言，突变往往发生在epoch = 150~300的位置。调小epoch后，发现32层的最优准确率甚至略高于16层，达到0.71（**如下图**）。当然，每次训练的结果均不尽相同，从平均值上看，准确率还是随层数增加而下降的。也因为突变的产生，表格中**32和64层的准确率均取峰值**。



<img src="Field3实验结果和问题回答.assets/f11ad7d1840178d89f01976c97874f3.jpg" alt="img" style="zoom:25%;" /><img src="Field3实验结果和问题回答.assets/896b5cc4a0f2a6d3245b5011cfce856.jpg" alt="img" style="zoom:25%;" />

<center><b>图3 修改Epoch后的32层GCN-Oringinal</b></center>





### 2.3 **DropEdge GCN的准确率随深度的变化**

类似地，对加入了DropEdge的GCN准确率进行探究。图4中，每层网络对应3对（6张）图片，从左至右依次是p = 0.3, 0.5, 0,7的训练结果。



![image-20221014094854893](Field3实验结果和问题回答.assets/image-20221014094854893.png)

<center><b>图4 训练结果-DropEdge</b></center>





| p值\网络层数 | 2     | 4     | 8     | 16    | 32            | 64            |
| ------------ | ----- | ----- | ----- | ----- | ------------- | ------------- |
| **0.3**      | 0.802 | 0.794 | 0.702 | 0.326 | 0.316（峰值） | 0.316（峰值） |
| **0.5**      | 0.796 | 0.753 | 0.761 | 0.668 | 0.341（峰值） | 0.316（峰值） |
| **0.7**      | 0.796 | 0.786 | 0.750 | 0.729 | 0.628（峰值） | 0.316（峰值） |

<center><b>表2 测试集准确率-DropEdge</b></center>





取DropEdge训练的最优准确率（不同层来自不同的p值)，与表1中的Original所得准确率进行比较，得到下面的表3。

| **网络层数**     | **2**  | **4**  | **8**  | **16** | **32**        | **64**        |
| ---------------- | ------ | ------ | ------ | ------ | ------------- | ------------- |
| **GCN-Original** | 0.796  | 0.788  | 0.771  | 0.694  | 0.710（峰值） | 0.391（峰值） |
| **GCN-DropEdge** | 0.802  | 0.794  | 0.761  | 0.729  | 0.628（峰值） | 0.316（峰值） |
| **增幅**         | +0.006 | +0.006 | -0.010 | +0.035 | -0.082        | -0.075        |

<center><b>表3 测试集准确率比较</b></center>





从表3中看出，DropEdge在16层网络及更浅的网络下，性能比Original好。在32层和64层，DropEdge的性能反而不如Original。同时，训练结果与论文提供的数据差异较大。总结几个可能的原因如下：

1. 训练集样本数量太少。随着网络深度增加，网络所产生的权重数目远远超过训练集样本数目。
   **拇指规则**指出，为了得到一个较好的泛化能力，我们需要满足以下条件（WidrowandStearns，1985；Haykin，2008）：N=nw/e. 其中，N为训练样本数量，nw是网络中突触权重的数量，e是测试允许的网络误差。因此，假如我们允许10%的误差，我们需要的训练样本的数量大约是网络中权重数量的10倍。显然在本次试验中，训练集没有满足这一条件，导致误差较大。

   对1的反驳：即使样本较少，但论文中的训练集与此相同，却得到了较好的结果？

   对1的反驳的反驳：下载论文源码并运行，16层时，Oringinal和DropEdge的准确率均在0.2左右浮动。到32层时，Original的准确率比DropEdge高6个百分点（如下图，上面是DropEdge，下面是Original）。对此，一个中肯的解释是，论文作者的超参数调的特别好。尽管如此，这说明论文的模型也存在这样的问题，并不是个例。

   ![image-20221014094909118](Field3实验结果和问题回答.assets/image-20221014094909118.png)

   ![image-20221014094916185](Field3实验结果和问题回答.assets/image-20221014094916185.png)

2. 由于时间有限，超参数未能调整至最优，

3. 论文模型采用了dropout机制，而本文中的模型没有dropout。



综上所述，DropEdge在浅层网络的优化性能确实存在。在32层及以上的网络深度，由于样本数目小、模型超参数未最优化等问题，暂时未能验证其优越性。



## 3 论文源码解读

与上述实验采用的模型相比，论文中的GCN层更加丰富，同时多加入了几个层，现解读如下：

### 1 输入层

```python
        if inputlayer == "gcn":
            # input gc
            self.ingc = GraphConvolutionBS(nfeat, nhid, activation, withbn, withloop)
            baseblockinput = nhid
        elif inputlayer == "none":
            self.ingc = lambda x: x
            baseblockinput = nfeat
        else:
            self.ingc = Dense(nfeat, nhid, activation)
            baseblockinput = nhid
```

输入层根据需要分为3种：

+ 以某种GCN层作为输入层
+ 直接传入数据，相当于一个一一映射
+ 采用了某种激活函数的全连接层

### 2 中间层

```python
        for i in range(nhidlayer):
            gcb = self.BASEBLOCK(in_features=baseblockinput,
                                 out_features=nhid,
                                 nbaselayer=nbaselayer,
                                 withbn=withbn,
                                 withloop=withloop,
                                 activation=activation,
                                 dropout=dropout,
                                 dense=False,
                                 aggrmethod=aggrmethod)
            self.midlayer.append(gcb)
            baseblockinput = gcb.get_outdim()
```

中间层铺设nhidlayer个GCN层，线性堆叠。其采用了dropout，随机失活部分神经元。



```python
if baseblock == "resgcn":
    self.BASEBLOCK = ResGCNBlock
elif baseblock == "densegcn":
    self.BASEBLOCK = DenseGCNBlock
elif baseblock == "mutigcn":
    self.BASEBLOCK = MultiLayerGCNBlock
elif baseblock == "inceptiongcn":
    self.BASEBLOCK = InecptionGCNBlock
```

这里的前提是事先声明的，GCN层的种类依据声明而定。

最基础的就是multigcn，多个线性叠加的GCN层，采用相同的输入和输出维度；

ResGCN则是在multigcn的基础上添加残差连接

DenseGCN是在计算输出时，每层的输出均包括前n层的输出，层层叠加。

InceptionGCN加入了Incept Block



### 3 输出层

```python
outactivation = lambda x: x  # we donot need nonlinear activation here.
self.outgc = GraphConvolutionBS(baseblockinput, nclass, outactivation, withbn, withloop)
```

输出层采用1层GCN层

