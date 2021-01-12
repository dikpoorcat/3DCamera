# # 以二进制方式读文件
# def read_b():
#     # f = open("e:/input.txt",'rb')
#     f = open(r"D:\oreo\photo\RulerXR 330\0.dat", 'rb')
#     print(f.read())
#     f.close()
#
#
# # 以二进制方式写入到文件
# def write_b():
#     f = open("e:/output.txt", "wb")
#     f.write(b'\xd6\xd0\xb9\xfa')
#     s = '\n中华人民共和国\n'
#     length = f.write(s.encode('utf-8'))
#     print("写入：{}-- 共写入：{}个字节".format(s, length))
#     f.close()
#
#
# # 主函数
# def main():
#     try:
#         read_b()
#         # write_b()
#     except IOError:
#         print("读取失败")
#
#
# main()


import numpy as np
a = 10*np.random.random((5, 5)) # 新建5*5矩阵做演示
print(a)
print('---')
amin, amax = a.min(), a.max() # 求最大最小值
a = (a-amin)/(amax-amin) # (矩阵元素-最小值)/(最大值-最小值)
print(a)