import pandas as pd
import pickle
import numpy as np
from tqdm import tqdm
import os


def get_data_with_windows(name = 'train'):
    with open(f'data/prepare/dict.pkl','rb') as f:
        map_dict = pickle.load(f)
    result = []
    root = os.path.join('data/prepare',name)
    files = list(os.listdir(root))

    for file in tqdm(files):
        path = os.path.join(root,file)
        samples = pd.read_csv(path,sep = ',')
        sep_index = [-1] + samples[samples['word']=='sep'].index.tolist()#-1 20 40 50
        for i in range(len(sep_index)):
            start = i + 1
            end = sep_index[i+1]

        print(sep_index)

if __name__ == '__main__':
    get_data_with_windows('train')