# A-MobileNetV2-based-Garbage-Classification-System-with-TensorFlow
# 1. 概述

## 简介

垃圾分类是当前社会关注的重要议题之一，传统的垃圾分类方式依靠人工识别，效率低下且容易出错。基于深度学习的垃圾分类技术能够有效解决上述问题，具有广阔的应用前景。

## 项目目标

本项目旨在利用 TensorFlow 框架构建一个垃圾分类系统，实现以下目标：

- 能够识别常见生活垃圾的类别；
- 具有较高的识别准确率；
- 可用于实际应用场景，如智能垃圾桶、垃圾分类小程序等。
![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/5291b363d4954e75902e642f5ddff704.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/c68445f367784bb8897811251d4719e1.png)

# 2. 环境配置

## TensorFlow 版本

本项目使用 TensorFlow 2.9.1 版本。

## 其他依赖库

NumPy
Matplotlib
Pillow
FastAPI
# 3. 数据集处理

## 数据集介绍

本项目使用 Kaggle 上的 垃圾分类数据集: [移除了无效网址]。该数据集包含 42 类垃圾的 12,000 张图像，每类图像 300 张。

## 数据预处理步骤

读取图像并转换为 RGB 格式；
缩放图像为统一大小（32x32 像素）；
将数据集划分为训练集和测试集，比例为 8:2；
将数据保存为 NumPy 数组格式。
## 加载预处理后的数据

```python
train_images = np.load("train_images.npy")
train_labels = np.load("train_labels.npy")
test_images = np.load("test_images.npy")
test_labels = np.load("test_labels.npy")
Use code with caution.
```

# 4. 模型构建

## 模型架构

本项目采用 MobileNetV2 作为模型架构，该模型具有较好的轻量性和准确性。

## 模型编译


```python
model = tf.keras.applications.MobileNetV2(
    input_shape=(32, 32, 3),
    weights="imagenet",
    include_top=False,
)

model.trainable = False

top_model = tf.keras.Sequential([
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(42, activation="softmax"),
])

model = tf.keras.Model(inputs=model.input, outputs=top_model(model.output))

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
```


# 5. 模型训练

## 训练过程


```python
model.fit(
    train_images,
    train_labels,
    epochs=10,
    batch_size=32,
    validation_data=(test_images, test_labels),
)
```


## 保存模型


```python
model.save("garbage_classification.h5")
Use code with caution.
```

# 6. 模型评估

## 测试集准确率


```python
loss, accuracy = model.evaluate(test_images, test_labels)
print("测试集准确率：", accuracy)
```


## 可视化预测结果


```python
import matplotlib.pyplot as plt

# 随机选取 10 张测试图像
test_images = test_images[:10]
test_labels = test_labels[:10]

# 预测结果
predictions = model.predict(test_images)

# 显示图像和预测结果
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(test_images[i])
    plt.title(f"预测：{label_dict[np.argmax(predictions[i])]}")
    plt.axis("off")

plt.show()
```


# 7. WEB应用部署

## FastAPI 简介

FastAPI 是一个现代、快速、高性能的 Python Web 框架。

## 构建简单表单


```html
<form action="/display_images" method="post">
    <label for="selected_label">选择分类标签：</label>
    <select name="selected_label" id="selected_label">
        {% for label in labels %}
        <option value="{{ label }}">{{ label }}</option>
        {% endfor %}
    </select>
```

    

# Sources

```txt
github.com/Berken-demirel/AI_works
github.com/voreille/2d_bispectrum_cnn subject to license (MIT)
github.com/Deepak3693/VQA_CoAttentionModel
```
# 项目链接

CSDN：[https://github.com/Olivia-account/A-MobileNetV2-based-Garbage-Classification-System-with-TensorFlow](https://blog.csdn.net/m0_46573428/article/details/136154843?csdn_share_tail=%7B%22type%22%3A%22blog%22%2C%22rType%22%3A%22article%22%2C%22rId%22%3A%22136154843%22%2C%22source%22%3A%22m0_46573428%22%7D)https://blog.csdn.net/m0_46573428/article/details/136154843?csdn_share_tail=%7B%22type%22%3A%22blog%22%2C%22rType%22%3A%22article%22%2C%22rId%22%3A%22136154843%22%2C%22source%22%3A%22m0_46573428%22%7D
# 后记
如果觉得有帮助的话，求 关注、收藏、点赞、星星 哦！
