## 中文说明
[https://github.com/TensorFlowNews/TensorFlow-Bitcoin-Robot/blob/master/README_CN.MD](https://github.com/TensorFlowNews/TensorFlow-Bitcoin-Robot/blob/master/README_CN.MD)

##  Into
A Bitcoin trade robot based on Tensorflow LSTM model.Just for fun.

##  DataSet
The data is got from https://api.btctrade.com/api/trades?coin=btc with requests.It includes 50 trades of Bitcoin.
get_trades.py will get the trades and show you with a picture.

![Figure_1.png](http://upload-images.jianshu.io/upload_images/76451-ebba6dc707ab1658.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

##  Model
rnn_predicter.py uses LSTM model.It use 10 trades as input ,if the next price is bigger than the 10st one ,the result is [1,0,0],if the next price is smaller than the 10st one ,the result is [0,0,1],if the next price is equal as 10st one ,the result is [0,1,0].

So,the [1,0,0] means that the price of Bitcoin will be higher.

##  Result
https://github.com/TensorFlowNews/TensorFlow-Bitcoin-Robot/blob/master/training_result.md

##  More
FaceRank - Rank Face by CNN Model based on TensorFlow (add keras version).
https://github.com/fendouai/FaceRank

## Blog
http://www.tensorflownews.com/

## Update

model saver

data saver

## WeChatGroup
![WeChatGroup Pic](https://github.com/fendouai/FaceRank/blob/master/wechatgroup.jpg)
