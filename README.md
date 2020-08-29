# 使用Tensorflow 


## 识别图片 - 介绍

[学习地址] (https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/?utm_campaign=chrome_series_machinelearning_063016&utm_source=gdev&utm_medium=yt-desc#0)

### 你将要构建什么

在这个codelab 中，你将学会如何在单机上运行Tensorflow，然后训练一个花的照片分类器。

我们将使用`transfer learning` 技术， 意思是说我们是从一个已经被训练好的模型中开始我们的训练。我们将训练一个相似的问题。深度学习从头开始训练会耗费好几天的时间，但是`transfer learning` 只需要很短时间呢。

我们将会使用一个在[ImageNet](http://image-net.org/) 大型视觉识别挑战的[数据集合](http://www.image-net.org/challenges/LSVRC/2012/)中。这个模型区分了1000 个不同的类型，例如斑点狗和洗碗机等。你可以有很多选择，你可以选择适合的，高效的精确的。

我们将使用同样的模型，但是重新训练根据我们的实例来进行。

### 你将会学到

如何使用Python 和Tensorflow训练一个图像分类器
如何使用训练好的分类器去给你的图片分类

### 你需要准备的

基本的理解Linux命令

## 设置

如何安装Tensorflow，[这里](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/?utm_campaign=chrome_series_machinelearning_063016&utm_source=gdev&utm_medium=yt-desc#1) 是原教程的方法，我直接使用docker镜像来运行。


	docker run -it -p 8888:8888 -p 6006:6006 tensorflow/tensorflow:latest-jupyter  # Start Jupyter server


## 下载训练图片
[源地址](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/?utm_campaign=chrome_series_machinelearning_063016&utm_source=gdev&utm_medium=yt-desc#2)

命令行下载图片

	curl http://download.tensorflow.org/example_images/flower_photos.tgz 

下载完后，将其移动到容器中去。

	docker cp ~/Downloads/flower_photos.tgz my_container_name:/root/tf_files

如果`root` 目录下没有`tf_files` 文件，请自行创建。


## 重新训练网络神经

这一节有很多坑，在这里直接给出我的解决方案, 注意新建一个`scripts/`然后将`retrain.py`放到里面。正常工作的[retrain.py](https://github.com/land-pack/tf/blob/master/retrain.py)




`python -m scripts.retrain  --image_dir=tf_files/flower_photos`


如果顺利的话，会在`tmp/`目录下生成`output_graph.pb ` 和 `output_labels.txt ` 文件。
将这两个文件复制到`tf_files/`目录下。



## 训练和仪表盘
我们暂时忽略，因为这个主要是调优的。


## 使用我们重新训练好的模型
先到这里把 [label_image.py](https://github.com/land-pack/tf/blob/master/label_image.py) 下载下来。

	python -m scripts.label_image  --input_layer="Placeholder" --output_layer=final_result --graph=tf_files/output_graph.pb --labels=tf_files/output_labels.txt --image=tf_files/flower_photos/daisy/21652746_cc379e0eea_m.jpg
	
运行完成可以查看到如下结果：

	2020-08-29 13:45:14.962201: W tensorflow/stream_executor/platform/default/dso_loader.cc:59] Could not load dynamic library 'libcudart.so.10.1'; dlerror: libcudart.so.10.1: cannot open shared object file: No such file or directory
	2020-08-29 13:45:14.962274: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
	WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow/python/compat/v2_compat.py:96: disable_resource_variables (from tensorflow.python.ops.variable_scope) is deprecated and will be removed in a future version.
	Instructions for updating:
	non-resource variables are not supported in the long term
	2020-08-29 13:45:17.138179: W tensorflow/stream_executor/platform/default/dso_loader.cc:59] Could not load dynamic library 'libcuda.so.1'; dlerror: libcuda.so.1: cannot open shared object file: No such file or directory
	2020-08-29 13:45:17.138290: W tensorflow/stream_executor/cuda/cuda_driver.cc:312] failed call to cuInit: UNKNOWN ERROR (303)
	2020-08-29 13:45:17.138341: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:156] kernel driver does not appear to be running on this host (6c0591a52d64): /proc/driver/nvidia/version does not exist
	2020-08-29 13:45:17.138740: I tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN)to use the following CPU instructions in performance-critical operations:  AVX2 FMA
	To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
	2020-08-29 13:45:17.147464: I tensorflow/core/platform/profile_utils/cpu_utils.cc:104] CPU Frequency: 2598050000 Hz
	2020-08-29 13:45:17.147785: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x7084a90 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
	2020-08-29 13:45:17.147845: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
	daisy 0.9974712
	sunflowers 0.0012744407
	dandelion 0.0006337628
	tulips 0.00049498084
	roses 0.0001255395


