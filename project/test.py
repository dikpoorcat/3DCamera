import cv2
import matplotlib.pyplot as plt
import numpy as np

np.set_printoptions(threshold=np.inf)


def dat2img(filename):
    # 从二进制数据中提取高度数据
    f = open(filename, 'rb')
    records = np.fromfile(f, dtype='<f', count=2560 * 600 * 5, offset=3000 * 2560)
    img = records.reshape((3000, -1))

    # 将负数替换成0
    img = img.clip(0)

    # range校正（仅适用于测试数据）
    colum1 = 700
    colum2 = 2000
    mean_arry = np.mean(img, axis=0)
    k = (mean_arry[colum2] - mean_arry[colum1]) / (colum2 - colum1)
    c = np.arange(0, k * 2560, k)
    img = img - c
    img_height = img.copy()
    # print(img_height[300, :])

    # 截取40高度位置
    img = img.clip(40)  # 背景：30  饼干：39~43mm

    # 归一化操作，*255当图片显示
    amin, amax = img.min(), img.max()  # 求最大最小值
    img = 255 * (img - amin) / (amax - amin)  # (矩阵元素-最小值)/(最大值-最小值)
    # plt.imshow(img)
    # plt.show()

    # 二值化处理
    # ret0, binary = cv2.threshold(img, 5, 255, cv2.THRESH_BINARY)
    # plt.imshow(binary)
    # plt.show()

    # resize image
    print('Original Dimensions : ', img.shape)
    height = 1560  # keep original height   1560比较合适
    width = img.shape[1]  # keep original width
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    print('Resized Dimensions : ', resized.shape)
    # plt.imshow(resized)
    # plt.show()
    img_height = cv2.resize(img_height, dim, interpolation=cv2.INTER_AREA)

    # 保存图片
    cv2.imwrite(r'H:\GitHub\3DCamera\docs\photos\resized.png', resized)
    cv2.imwrite(r'H:\GitHub\3DCamera\docs\photos\img_height.png', img_height)

    return resized, img_height


def dimensionMeasurement(img, img_height):
    img_original = img.copy()
    img_uint8 = img.astype(np.uint8)  # 转换为整型
    img_info = img.shape
    image_height = img_info[0]
    image_weight = img_info[1]
    mask1 = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)

    # 二值化处理
    ret0, binary = cv2.threshold(img_uint8, 5, 255, cv2.THRESH_BINARY)
    # plt.imshow(binary, cmap='gray')
    # plt.show()

    # 开闭操作
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    # plt.imshow(binary, cmap='gray')
    # plt.show()
    cv2.imwrite(r'H:\GitHub\3DCamera\docs\photos\binary.png', binary)

    # 寻找最外面的图像轮廓 返回修改后的图像 图像的轮廓  以及它们的层次
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # print('type(contours):', type(contours))
    # print('type(contours[0]):', type(contours[0]))
    print('len(contours):', len(contours))

    # 转换为RGB图
    img = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    img[:, :, 0] = img_uint8
    img[:, :, 1] = img_uint8
    img[:, :, 2] = img_uint8
    # plt.imshow(img)
    # plt.show()

    # 遍历每一个轮廓
    for c in contours:
        # # 找到边界框的坐标
        # x, y, w, h = cv2.boundingRect(c)
        # # 在img图像上 绘制矩形  线条颜色为green 线宽为2
        # cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),1)
        #
        # # 找到最小区域
        # rect = cv2.minAreaRect(c)
        #
        # # 计算最小矩形的坐标
        # box = cv2.boxPoints(rect)
        #
        # # 坐标转换为整数
        # box = np.int0(box)
        #
        # # 绘制轮廓  最小矩形 blue
        # cv2.drawContours(img,[box],0,(255,0,0),3)

        # 计算闭圆中心点和半径
        (x, y), radius = cv2.minEnclosingCircle(c)

        # 转换为整型
        center = (int(x), int(y))
        radius = int(radius)
        # print("中心及半径", center, radius)

        # 在mask1绘制闭圆
        mask1 = cv2.circle(mask1, center, radius, (255, 255, 255), 1)

        # 在img绘制闭圆
        img = cv2.circle(img, center, radius, (0, 0, 255), 1)
        # 绘制圆心
        #     img = cv2.circle(img,center,5,(0,0,255),-1)
        # 在圆心绘制十字
        r = 7
        cv2.line(img, (int(x) - r, int(y)), (int(x) + r, int(y)), (0, 0, 255), 2)
        cv2.line(img, (int(x), int(y) - r), (int(x), int(y) + r), (0, 0, 255), 2)
        # 画出直径
        #     cv2.line(img,(int(x)-radius,int(y)),(int(x)+radius,int(y)),(0,0,255),2)
        # 标出直径
        pixelsPerMetric = 2560 / 390.683895
        d = radius * 2 / pixelsPerMetric
        cv2.putText(img, "diameter: {:.2f}mm".format(d),
                    (int(x - 100), int(y - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (50, 100, 255), 2)
        # 标出高度
        height = 0
        k = 10
        for i in range(k):  # 取k*k范围内像素的平均高度
            for j in range(k):
                height += img_height[int(y + i), int(x + j)] - 31.4
                # print(height)
        height /= (k * k)

        # height = img_height[int(y), int(x)] - 31.4
        cv2.putText(img, "height: {:.2f}mm".format(height),
                    (int(x - 72), int(y - 30)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (50, 100, 255), 2)

    # 获取圆内mask（圆内数据有效）
    maskInside = cv2.cvtColor(mask1, cv2.COLOR_BGR2GRAY)
    seedPoint = (0, 0)
    newVal = 255
    mask = np.zeros([image_height + 2, image_weight + 2], np.uint8)  # 新建图像矩阵  +2是官方函数要求
    cv2.floodFill(maskInside, mask, seedPoint, newVal)
    maskInside = ~maskInside
    # plt.imshow(maskInside, cmap='gray')
    # plt.show()

    # 获取轮廓外mask
    maskOutside = ~binary.astype(np.uint8)
    # plt.imshow(maskOutside, cmap='gray')
    # plt.show()

    # 获取缺损处mask
    mask = ~((maskInside + maskOutside) == 255)
    # plt.imshow(mask, cmap='gray')
    # plt.show()

    # 在img上填充颜色
    output = img.copy()
    output[:, :, 0][mask] = np.random.randint(0, 255)  # mask部分会随机填充颜色
    output[:, :, 1][mask] = np.random.randint(0, 255)  # mask部分会随机填充颜色
    output[:, :, 2][mask] = np.random.randint(0, 255)  # mask部分会随机填充颜色
    # cv2.imshow('output',output)

    # 在img上绘制轮廓
    cv2.drawContours(output, contours, -1, (0, 0, 255), 1)
    cv2.imshow('contours', output)

    cv2.waitKey()
    cv2.destroyAllWindows()
    cv2.imwrite(r'H:\GitHub\3DCamera\docs\photos\@output.png', output)
    print('尺寸检测完成。')


def patternMeasurement(img):
    img_original = img.copy()
    img_uint8 = img.astype(np.uint8)  # 转换为整型
    img_info = img.shape
    image_height = img_info[0]
    image_weight = img_info[1]

    # 二值化处理:自适应，阈值取相邻区域的平均值
    adaptive1 = cv2.adaptiveThreshold(img_uint8, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    # cv2.imshow('adaptive1', adaptive1)
    # cv2.waitKey(0)
    # plt.imshow(adaptive1, cmap='gray')
    # plt.show()

    # floodfill背景
    seedPoint = (0, 0)
    newVal = 0
    loDiff = 15
    upDiff = 55
    mask = np.zeros([image_height + 2, image_weight + 2], np.uint8)  # 新建图像矩阵  +2是官方函数要求
    cv2.floodFill(adaptive1, mask, seedPoint, newVal, loDiff, upDiff, cv2.FLOODFILL_FIXED_RANGE)

    # 开操作：去除白噪点
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
    adaptive1 = cv2.morphologyEx(adaptive1, cv2.MORPH_OPEN, kernel)
    # plt.imshow(binary, cmap='gray')
    # plt.show()
    cv2.imwrite(r'H:\GitHub\3DCamera\docs\photos\adaptive1.png', adaptive1)

    # 转换为RGB图
    img = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    img[:, :, 0] = img_uint8
    img[:, :, 1] = img_uint8
    img[:, :, 2] = img_uint8
    # plt.imshow(img)
    # plt.show()

    # 绘制轮廓
    contours, hierarchy = cv2.findContours(adaptive1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (0, 0, 255), 1)
    print('len(contours):', len(contours))
    cv2.imshow("img-contours", img)
    cv2.waitKey(0)
    cv2.imwrite(r'H:\GitHub\3DCamera\docs\photos\contours.png', img)
    print('图案检测完成。')


if __name__ == '__main__':
    img, img_height = dat2img(r'H:\GitHub\3DCamera\docs\data\0.dat')
    dimensionMeasurement(img, img_height)
    patternMeasurement(img)
    print('程序运行结束。')
