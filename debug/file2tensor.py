import re
import numpy as np

class ParseBlob:
    def __init__(self, filename):
        self.blobfile = filename
        self.info_line = None
        self.data_line = None
        self.shape = None
        self.array = None
        self.readblob()
        self.internal_get_shape()
        self.internal_get_tensor()

    def readblob(self):
        with open(self.blobfile, 'r') as file:
            # 逐行读取文件内容并存放在一个列表中
            lines = file.readlines()
        self.info_line = lines[0]
        self.data_line = lines[1:]

    def internal_get_shape(self):
        info_string = self.info_line

        pattern = r': (\d+(?: \d+)*) \('
        shape_numbers = re.search(pattern, info_string)

        if shape_numbers:
            numbers_str = shape_numbers.group(1)  # 获取匹配的数字字符串
            self.shape = [int(num) for num in numbers_str.split()]  # 将数字字符串拆分为列表
            print(self.shape)
        else:
            print("没有找到匹配的数字列表")
            raise RuntimeError("Not match list")

    def internal_get_tensor(self):
        cvt_list = [float(item) for item in self.data_line]
        np_array = np.array(cvt_list)
        self.array = np_array.reshape(self.shape)

    def get_shape(self):
        return self.shape

    def get_tensor(self):
        return self.array

if __name__ == "__main__":
    blob = "#443_Convolution_L0104_Conv2d_BN_WithoutBiases_out0.ieb"
    parser = ParseBlob(blob)
    import pdb; pdb.set_trace()
    shape = parser.get_shape()
    value = parser.get_tensor()
    print(shape)
    print(value)
