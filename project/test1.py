import struct

list_dec = [1, 2, 3, 4, 53, 100, 220, 244, 255]
with open('hexBin.dat', 'wb')as fp:
    for x in list_dec:
        a = struct.pack('B', x)
        fp.write(a)

print('done')
f = open("hexBin.bin", 'rb')
print(f.read())