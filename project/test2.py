# import struct
#
# # rb 表示以二进制形式打开文件
# with open(r"D:\oreo\photo\RulerXR 330\0.dat", mode="rb") as f:
#     # 移至指定字节位置
#     f.seek(416)
#     # 读入 16 个字节
#     a = f.read(16)
#     # 打印 a 类型 bytes
#     print(type(a))
#     # 打印 a 内字节数目
#     print(len(a))
#     # 打印 a 内数据，以 16 进制数显示
#     print(a)
#     # 16 个字节解析为 4 个 unsigned short 数据和 2 个 unsigned int 数据，字节排序为小端，返回元组
#     val_tuple = struct.unpack("<4H2I", a) # 如果解析 1 个数据，则应当读取与数据存储空间大小一致的字节数目，unpack 仍返回元组
#     print(val_tuple)
#     # 将元组转为 list
#     val_list = list(val_tuple)
#     print(val_list)

import struct
from struct import Struct

# # rb 表示以二进制形式打开文件
# with open(r"D:\oreo\photo\RulerXR 330\0.dat", mode="rb") as f:
#     # for i in range():
#
#
#
#     # 移至指定字节位置
#     f.seek(416)
#     # 读入 16 个字节
#     a = f.read(10)
#     # 打印 a 类型 bytes
#     # print(type(a))
#     # 打印 a 内字节数目
#     print(len(a))
#     # 打印 a 内数据，以 16 进制数显示
#     print(a)
#     # 16 个字节解析为 4 个 unsigned short 数据和 2 个 unsigned int 数据，字节排序为小端，返回元组
#     val_tuple = struct.unpack("<fBfB", a) # 如果解析 1 个数据，则应当读取与数据存储空间大小一致的字节数目，unpack 仍返回元组
#     print(val_tuple)
#     # 将元组转为 list
#     val_list = list(val_tuple)
#     print(val_list)



# def unpack_records(format, data):
#   record_struct = Struct(format)
#   return (record_struct.unpack_from(data, offset)
#       for offset in range(0, len(data), record_struct.size))
#
# # Example
# if __name__ == '__main__':
#   with open(r'D:\oreo\photo\RulerXR 330\0.dat', 'rb') as f:
#     data = f.read(500)
#   for rec in unpack_records('<fB', data):
#     # Process rec
#     print(rec)

# def read_records(format, f):
#   record_struct = Struct(format)
#   chunks = iter(lambda: f.read(record_struct.size), b'')
#   return (record_struct.unpack(chunk) for chunk in chunks)
#
# # Example
# if __name__ == '__main__':
#   with open(r'D:\oreo\photo\RulerXR 330\0.dat','rb') as f:
#     for rec in read_records('<fB', f):
#       # Process rec
#       print(rec)

import numpy as np
import cv2

for i in range(5):
    f = open(r'D:\oreo\photo\RulerXR 330\0.dat', 'rb')
    records = np.fromfile(f, dtype='<f, B', count=2560*600*(5-i), offset=(600*2560+4)*i)
    print(records)
    img = records['f1'].reshape((600 * (5-i), -1))
    print(img)
    name = "D:\\opencv\\records"+str(i)+".png"
    cv2.imwrite(name, img)


# print(records)
# print(records.dtype)
# print(records.shape)

# print(records['f1'])
# print(records['f1'].dtype)
# print(records['f1'].shape)



# x = np.array([(1.0, 4.0,), (2.0, -1.0)], dtype=[('f0', '<f8'), ('f1', '<f8')])
# print(x)
# x = x.view(np.float).reshape(x.shape + (-1,))
# print(x)

