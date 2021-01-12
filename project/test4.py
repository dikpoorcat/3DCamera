import numpy as np
import cv2

# for i in range(5):
#     f = open(r'D:\oreo\photo\RulerXR 330\0.dat', 'rb')
#     records = np.fromfile(f, dtype='<f, B', count=2560*600, offset=(600*2560)*i)
#     print(records)
#     img = records['f1'].reshape((600, -1))
#     print(img)
#     name = "D:\\opencv\\test"+str(i)+".png"
#     cv2.imwrite(name, img)

img = np.empty((0, 2560))
# img = np.array([]).reshape((0, 2560))

for i in range(3000):
    # 读取1行，是5张并排的
    f = open(r'D:\oreo\photo\RulerXR 330\0.dat', 'rb')
    records = np.fromfile(f, dtype='<f, B', count=2560*1, offset=2560*i)
    # print(records)
    # print(records.shape)

    # reshape成1张的宽度
    step1 = records['f1'].reshape(-1, 512)
    # print(step1)
    # cv2.imshow('step1', step1)

    # 转置
    step2 = step1.T
    # print(step2)
    # cv2.imshow('step2', step2)

    # reshape回1行
    step3 = step2.reshape(1, -1)
    # print(step3)
    # cv2.imshow('step3', step3)

    # 拼接图像
    img = np.vstack((img, step3))
    # print("img: ", img)
    # print("img.shape: ", img.shape)

# cv2.imshow('img', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
cv2.imwrite("D:\\opencv\\test4.png", img)

