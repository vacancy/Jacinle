import numpy as np
from logic_induction import search
from sklearn import tree
import torch

# inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [1, 1], [1, 1], [1, 1]], dtype='uint8')
# outputs = np.array([[0], [1], [1], [1], [0], [0], [0]], dtype='uint8')
#
# input_names = ['x', 'y']
# dt = tree.DecisionTreeClassifier()
# # print(search(inputs, outputs, input_names))
# dt.fit(inputs, outputs)
# t = dt.tree_


if __name__ == '__main__':
    data = {'01110': '0', '01111': '1', '00000': '1', '01000': '1', '00010': '1', '01010': '0', '01100': '0',
            '11110': '1',
            '10010': '1', '11111': '1', '01011': '0', '11010': '0', '00100': '0', '00110': '0', '10110': '1',
            '11011': '0'}
    input_list = []
    output_list = []
    for k, v in data.items():
        input_list.append([int(x) for x in k])
        output_list.append([int(v)])
    name_list = ['abcdefghijklmn'[i] for i in range(len(input_list[0]))]
    print(233)
    print(search(np.array(input_list, dtype='uint8'), np.array(output_list, dtype='uint8'), name_list, type='general', coverage=0.9))
    exit()
