# import paddle.vision.transforms as T
# import paddle
# import numpy as np
# from paddle.vision.transforms import Resize
# transform = T.Compose([T.Grayscale(3),T.Normalize(mean=[127.5],
#                                    std=[127.5],
#                                    data_format='HWC'),
#                                    T.Resize(size=[224,224]),
#                                    T.ToTensor()])
# # 使用transform对数据集做归一化
# print('Start download training data and load training data.')

# # 加载FashionMNIST数据集
# train_dataset = paddle.vision.datasets.FashionMNIST(mode='train', transform=transform)
# test_dataset = paddle.vision.datasets.FashionMNIST(mode='test', transform=transform)
# print('Finished.')


# # GoogLeNet模型代码
# import numpy as np
# import paddle
# from paddle.nn import Conv2D, MaxPool2D, AdaptiveAvgPool2D, Linear
# ## 组网
# import paddle.nn.functional as F

# import numpy as np
# import paddle
# from paddle.nn import Conv2D, MaxPool2D, AdaptiveAvgPool2D, Linear
# ## 组网
# import paddle.nn.functional as F

# # 定义Inception块
# class Inception(paddle.nn.Layer):
#     def __init__(self, c0, c1, c2, c3, c4, **kwargs):
#         '''
#         Inception模块的实现代码，
        
#         c1,图(b)中第一条支路1x1卷积的输出通道数，数据类型是整数
#         c2,图(b)中第二条支路卷积的输出通道数，数据类型是tuple或list, 
#                其中c2[0]是1x1卷积的输出通道数，c2[1]是3x3
#         c3,图(b)中第三条支路卷积的输出通道数，数据类型是tuple或list, 
#                其中c3[0]是1x1卷积的输出通道数，c3[1]是3x3
#         c4,图(b)中第一条支路1x1卷积的输出通道数，数据类型是整数
#         '''
#         super(Inception, self).__init__()
#         # 依次创建Inception块每条支路上使用到的操作
#         self.p1_1 = Conv2D(in_channels=c0,out_channels=c1, kernel_size=1, stride=1)
#         self.p2_1 = Conv2D(in_channels=c0,out_channels=c2[0], kernel_size=1, stride=1)
#         self.p2_2 = Conv2D(in_channels=c2[0],out_channels=c2[1], kernel_size=3, padding=1, stride=1)
#         self.p3_1 = Conv2D(in_channels=c0,out_channels=c3[0], kernel_size=1, stride=1)
#         self.p3_2 = Conv2D(in_channels=c3[0],out_channels=c3[1], kernel_size=5, padding=2, stride=1)
#         self.p4_1 = MaxPool2D(kernel_size=3, stride=1, padding=1)
#         self.p4_2 = Conv2D(in_channels=c0,out_channels=c4, kernel_size=1, stride=1)
        
#         # # 新加一层batchnorm稳定收敛
#         # self.batchnorm = paddle.nn.BatchNorm2D(c1+c2[1]+c3[1]+c4)

#     def forward(self, x):
#         # 支路1只包含一个1x1卷积
#         p1 = F.relu(self.p1_1(x))
#         # 支路2包含 1x1卷积 + 3x3卷积
#         p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
#         # 支路3包含 1x1卷积 + 5x5卷积
#         p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
#         # 支路4包含 最大池化和1x1卷积
#         p4 = F.relu(self.p4_2(self.p4_1(x)))
#         # 将每个支路的输出特征图拼接在一起作为最终的输出结果
#         return paddle.concat([p1, p2, p3, p4], axis=1)
#         # return self.batchnorm()
    
# class GoogLeNet(paddle.nn.Layer):
#     def __init__(self):
#         super(GoogLeNet, self).__init__()
#         # GoogLeNet包含五个模块，每个模块后面紧跟一个池化层
#         # 第一个模块包含1个卷积层
#         self.conv1 = Conv2D(in_channels=3,out_channels=64, kernel_size=7, padding=3, stride=1)
#         # 3x3最大池化
#         self.pool1 = MaxPool2D(kernel_size=3, stride=2, padding=1)
#         # 第二个模块包含2个卷积层
#         self.conv2_1 = Conv2D(in_channels=64,out_channels=64, kernel_size=1, stride=1)
#         self.conv2_2 = Conv2D(in_channels=64,out_channels=192, kernel_size=3, padding=1, stride=1)
#         # 3x3最大池化
#         self.pool2 = MaxPool2D(kernel_size=3, stride=2, padding=1)
#         # 第三个模块包含2个Inception块
#         self.block3_1 = Inception(192, 64, (96, 128), (16, 32), 32)
#         self.block3_2 = Inception(256, 128, (128, 192), (32, 96), 64)
#         # 3x3最大池化
#         self.pool3 = MaxPool2D(kernel_size=3, stride=2, padding=1)
#         # 第四个模块包含5个Inception块
#         self.block4_1 = Inception(480, 192, (96, 208), (16, 48), 64)
#         self.block4_2 = Inception(512, 160, (112, 224), (24, 64), 64)
#         self.block4_3 = Inception(512, 128, (128, 256), (24, 64), 64)
#         self.block4_4 = Inception(512, 112, (144, 288), (32, 64), 64)
#         self.block4_5 = Inception(528, 256, (160, 320), (32, 128), 128)
#         # 3x3最大池化
#         self.pool4 = MaxPool2D(kernel_size=3, stride=2, padding=1)
#         # 第五个模块包含2个Inception块
#         self.block5_1 = Inception(832, 256, (160, 320), (32, 128), 128)
#         self.block5_2 = Inception(832, 384, (192, 384), (48, 128), 128)
#         # 全局池化，用的是global_pooling，不需要设置pool_stride
#         self.pool5 = AdaptiveAvgPool2D(output_size=1)
#         self.fc = Linear(in_features=1024, out_features=1)

#     def forward(self, x):
#         x = self.pool1(F.relu(self.conv1(x)))
#         x = self.pool2(F.relu(self.conv2_2(F.relu(self.conv2_1(x)))))
#         x = self.pool3(self.block3_2(self.block3_1(x)))
#         x = self.block4_3(self.block4_2(self.block4_1(x)))
#         x = self.pool4(self.block4_5(self.block4_4(x)))
#         x = self.pool5(self.block5_2(self.block5_1(x)))
#         x = paddle.reshape(x, [x.shape[0], -1])
#         x = self.fc(x)
#         return x

# # 创建模型
# model = GoogLeNet()
# print(len(model.parameters()))
# opt = paddle.optimizer.Momentum(learning_rate=0.001, momentum=0.9, parameters=model.parameters(), weight_decay=0.001)
# # 启动训练过程
# model = paddle.Model(model)
# model.prepare(paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters()), 
#               paddle.nn.CrossEntropyLoss(), 
#               paddle.metric.Accuracy())
# model.fit(train_dataset, epochs=50, batch_size=64, verbose=1)

import os
import re
import shutil
# classification_modules = ['vgg19_imagenet', 'xception71_imagenet', 'xception41_imagenet', 'resnext50_vd_32x4d_imagenet', 'alexnet_imagenet',
#                         'mobilenet_v2_imagenet', 'se_resnext101_32x4d_imagenet', 'shufflenet_v2_imagenet', 'mobilenet_v3_large_imagenet_ssld',
#                         'vgg16_imagenet', 'vgg13_imagenet', 'resnet_v2_18_imagenet', 'resnext101_64x4d_imagenet',
#                         'inception_v4_imagenet', 'resnext50_32x4d_imagenet', 'resnext152_64x4d_imagenet', 'resnext50_64x4d_imagenet', 'densenet169_imagenet']

def main():
    classification_modules = []
    with open('allclassificationmodelname.txt', 'r') as namefile:
        modulenames = namefile.readlines()
        for modulename in modulenames:
            classification_modules.append(modulename.strip())
    print(classification_modules)

    for modulename in classification_modules:
        filename = os.path.join('modules/image/classification', modulename, 'README.md')
        newfilename = os.path.join('modules/image/classification', modulename, 'NewREADME.md')
        # with open(filename, 'r') as fp:
        #     with open(newfilename, 'w') as newfp:
        #         for line in fp.readlines():
        #             if re.search('文字识别', line):
        #                 line = line.replace('文字识别', '图像分类')
        #             newfp.write(line)
        # shutil.move(newfilename, filename)
        # with open(filename, 'r') as fp:
        #     with open(newfilename, 'w') as newfp:
        #         for line in fp.readlines():
        #             if re.search('应用效果展示', line):
        #                 continue
        #             if re.search('样例结果示例', line):
        #                 continue
        #             newfp.write(line)
        # with open(filename, 'r') as fp:
        #     with open(newfilename, 'w') as newfp:
        #         for line in fp.readlines():
        #             if re.search('PATH/TO/IMAGE', line):
        #                 if '/PATH/TO/IMAGE' in line:
        #                     pass
        #                 else:
        #                     line = line.replace('PATH/TO/IMAGE', '/PATH/TO/IMAGE')
        #             if '代码示例' in line:
        #                 line = line.replace('代码示例', '预测代码示例')
        #             newfp.write(line)
        # with open(filename, 'r') as fp:
        #     with open(newfilename, 'w') as newfp:
        #         for line in fp.readlines():
        #             if '|是否支持Fine-tuning|是|' in line:
        #                 line = line.replace('|是否支持Fine-tuning|是|', '|是否支持Fine-tuning|否|')
        #             newfp.write(line)
        # shutil.move(newfilename, filename)
        with open(filename, 'r') as fp:
            with open(newfilename, 'w') as newfp:
                for line in fp.readlines():
                    
                    if "```python" in line:
                        line = line.replace('|是否支持Fine-tuning|是|', '|是否支持Fine-tuning|否|')
                    newfp.write(line)
        shutil.move(newfilename, filename)
        
main()
        
        
# def main():
#     modules = ['resnext101_32x48d_wsl', 'dpn68_imagenet', 'spinalnet_vgg16_gemstone', 'food_classification',
#     'resnext101_32x32d_wsl', 'densenet201_imagenet', 'efficientnetb0_imagenet', 'resnext101_32x8d_wsl',
#     'efficientnetb3_imagenet', 'dpn131_imagenet', 'densenet264_imagenet', 'spinalnet_res50_gemstone', 'dpn98_imagenet']
#     for modulename in modules:
#         os.system("hub install {} && du -h ~/.paddlehub/modules/{} | awk 'END {{print}}' >> modulesize.txt".format(modulename, modulename))
        
# main()