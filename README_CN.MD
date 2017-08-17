##  简介
一个比特币交易机器人基于 Tensorflow LSTM 模型，仅供娱乐。
A Bitcoin trade robot based on Tensorflow LSTM model.Just for fun.

##  数据集
数据来自 btctrade ，用 requests 爬取，它包含比特币的 50 个交易记录。
get_trades.py 会获取这些交易记录并且用图片的方式展示出来。

![Figure_1.png](http://upload-images.jianshu.io/upload_images/76451-ebba6dc707ab1658.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

##  模型
rnn_predicter.py 使用  LSMT 模型。截取 10个交易记录作为输入，如果 第 11个加个比第10个高，就把输出设置为 [1,0,0]，如果低就设置为 [0,0,1] ，如果相同 [0,1,0]。

##  训练结果
https://github.com/TensorFlowNews/TensorFlow-Bitcoin-Robot/blob/master/training_result.md

##  之前的项目推荐
FaceRank - Rank Face by CNN Model based on TensorFlow (add keras version).
https://github.com/fendouai/FaceRank

## 后续更新发布
http://www.tensorflownews.com/

## 更新计划
模型持久化，训练数据集持久化。

## 微信群
![https://github.com/TensorFlowNews/TensorFlow-Bitcoin-Robot/blob/master/wechatgroup.jpg](https://github.com/TensorFlowNews/TensorFlow-Bitcoin-Robot/blob/master/wechatgroup.jpg)

## QQ群：
522785813