import pandas as pd
import pickle
import numpy
from tqdm import tqdm
import os


def get_data_with_windows(name = 'train'):
    with open(f'data/prepare/dict.pkl','rb') as f:
        map_dict = pickle.load(f)

    def item2id(data,w2i):
        # print(data)
        # count = 0
        return [w2i[x] if x in w2i else w2i['UNK'] for x in data ]

    results = []
    root = os.path.join('data/prepare/',name )
    files = list(os.listdir(root))

    for file in tqdm(files):
        result = []
        path = os.path.join(root,file)
        samples = pd.read_csv(path,sep = ',')
        num_samples = len(samples)
        sep_index = [-1] + samples[samples['word']=='sep'].index.tolist() + [num_samples]
        #----------------获取句子并将句子全部转换为id-------------------------
        for i in range(len(sep_index)-1):
            start = sep_index[i] + 1
            end = sep_index[i+1]
            data = []
            for feature in samples.columns:
                data.append(item2id(list(samples[feature])[start:end],map_dict[feature][1]))
            result.append(data)
        #----------------数据增强--------------------------------------------
        two = []
        for i in range(len(result)-1):
            first = result[i]
            second =result[i+1]
            two.append(first[k]+second[k] for k in range(len(first)))
        three = []
        for i in range(len(result)-2):
            first = result[i]
            second =result[i+1]
            third = result[i+2]
            three.append(first[k]+second[k]+third[k] for k in range(len(first)))
        results.extend(result + two + three)
    return results

if __name__ == '__main__':
    print(get_data_with_windows('train'))