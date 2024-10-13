# -*- coding: utf-8 -*-
# @Time    : 2024/9/2 下午11:04
# @Author  : Chen Mukun
# @File    : mapping.py
# @Software: PyCharm
# @desc    : 
def get_mapping(name):
    assert name in ["drugbank", "twosides"]
    id_smi = {}
    id_desc = {}
    if name == "drugbank":
        with open("./data/drugbank.tab", "r") as f:
            for i in f:
                if not i.startswith("\""):
                    continue
                data = i.strip().split('\t')
                assert len(data) == 6
                id_smi[data[0].replace("\"", "")] = data[4].replace("\"", "")
                id_smi[data[1].replace("\"", "")] = data[5].replace("\"", "")
                id_desc[int(data[2]) - 1] = data[3].replace("\"", "")
            f.close()
    else:
        with open("./data/twosides_ge_500.csv", "r") as f:
            for i in f:
                if not i.startswith("CID"):
                    continue
                data = i.strip().split(',')
                if len(data) != 6:
                   continue
                id_smi[data[0]] = data[1]
                id_smi[data[2]] = data[3]
            f.close()
        with open("./data/twosides_side_effect_info.csv", "r") as f:
            for i in f:
                data = i.strip().split(',')
                if not data[1].startswith("C"):
                    continue
                assert len(data) == 5
                id_desc[int(data[2])] = data[4]
            f.close()
    return id_smi, id_desc


if __name__ == '__main__':
    # print(get_mapping("drugbank"))
    print(get_mapping("1"))