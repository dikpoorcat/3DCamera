import numpy as np

np.set_printoptions(threshold=np.inf)
import cv2
import matplotlib.pyplot as plt
from skimage import data, exposure
from sklearn.preprocessing import normalize

# # 灰度数据
# f = open(r'D:\oreo\photo\RulerXR 330\3.dat', 'rb')
# records = np.fromfile(f, dtype='B', count=2560*600*5, offset=(600*2560)*0)
# img = records.reshape((3000, -1))
# # print(img)
#
# cv2.imwrite("D:\\opencv\\test6_3.png", img)


# 高度数据
f = open(r'D:\oreo\photo\RulerXR 330\0.dat', 'rb')
records = np.fromfile(f, dtype='<f', count=2560 * 600 * 5, offset=3000 * 2560)
img = records.reshape((3000, -1))

# 将负数替换成0
img_0 = img.clip(0)
img = img.clip(0)

# range校正
colum1 = 700
colum2 = 2000
mean_arry = np.mean(img, axis=0)
k = (mean_arry[colum2] - mean_arry[colum1]) / (colum2 - colum1)
# print(mean_arry[colum1], mean_arry[colum2], k)
c = np.arange(0, k * 2560, k)
img = img - c
img = img.clip(40)  #背景：30  饼干：39~43

# 归一化操作，*255当图片显示
amin, amax = img.min(), img.max()  # 求最大最小值
img = 255 * (img - amin) / (amax - amin)  # (矩阵元素-最小值)/(最大值-最小值)

# 保存图片
cv2.imwrite("D:\\opencv\\test6_hight.png", img)
print(img[500, :])

plt.imshow(img, plt.cm.gray)  # 均衡化图像
plt.show()


















# 均衡化
# img = exposure.equalize_hist(img)
# print(img[2999, :])
# print(img[:, 2000])


# plt.figure(figsize=(20, 10))
# plt.subplot(1, 2, 1)  # 将画板分为2行两列，本幅图位于第一个位置
# plt.hist(img)
# plt.title("histogram")
# plt.subplot(1, 2, 2)  # 将画板分为2行两列，本幅图位于第一个位置
# plt.hist(img_0)
# plt.title("histogram_img_0")
# plt.show()

# plt.hist(img)
# plt.title("histogram")
# plt.show()
