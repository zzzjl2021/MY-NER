import os
import pandas as pd
from collections import Counter
from data_process import split_text
from tqdm import tqdm
import jieba.posseg as psg
from cnradical import Radical,RunOption
import shutil #创建多层目录
from random import shuffle
import pickle

train_dir = './ruijin_round1_train2_20181022'

def process_text(idx,split_method = None,split_name = 'train'):
    """
    读取文本 切割然后打上标记 并提取词边界、词性、偏旁部首、拼音等文本特征
    :param: idx 文件的名字，不含扩展名
    :param: split_method 切割文本的方法，是一个函数
    :param: split_name 最终保存的文件夹名字

    :return:
    """
    data = {}
    #-------------------获取句子------------------------------------
    if split_method == None:
        with open(f'{train_dir}/{idx}.txt','r',encoding='utf8') as f:
            texts =f.readlines()
    else:
        with open(f'{train_dir}/{idx}.txt','r',encoding='utf8') as f:
            texts = f.read()
            texts = split_method(texts)
    data['word'] = texts
    #--------------------获取标签--------------------------------------------
    tag_list = ['O' for s in texts for x in s]
    tag = pd.read_csv(f'{train_dir}/{idx}.ann',header=None,sep='\t')
    for i in range(tag.shape[0]):
        tag_item = tag.iloc[i][1].split(' ') #获取实体的类别以及起始位置
        cls,start,end = tag_item[0],int(tag_item[1]),int(tag_item[-1]) #cls是class实体类别 转换成对应的类型
        tag_list[start] = 'B-'+cls 
        for j in range(start+1,end):
            tag_list[j] = 'I-'+cls
    assert len([x for s in texts for x in s])==len(tag_list) #保证两个序列的长度一致

#---------------------提取词性和词边界特征--------------------------------
    word_bounds = ['M' for item in tag_list] #首先给所有的字标上M记号
    word_flags = [] #保存词性 
    for text in texts:
        for word , flag in psg.cut(text):
            if len(word) == 1:
                start = len(word_flags) #拿到起始下标
                word_bounds[start] = 'S' 
                word_flags.append(flag) # 将当前的词性名加入wordflags列表
            else:
                start = len(word_flags)
                word_bounds[start] = 'B'
                word_flags += [flag] * len(word)            
                end = len(word_flags) - 1
                word_bounds[end] = 'E'

#---------------------------统一截断-----------------------------------
    tags = []
    bounds = []
    flags = []
    start = 0
    end = 0
    for s in texts:
        l = len(s)
        end =+ l
        bounds.append(word_bounds[start:end])
        flags.append(word_flags[start:end])
        tags.append(tag_list[start:end])
        start += 1
    data['bounds'] = bounds
    data['flag'] = flags
    data['label'] = tags

#-----------------获取拼音特征------------------------------
    radical = Radical(RunOption.Radical) #提取偏旁部首
    pinyin = Radical(RunOption.Pinyin) #提取偏旁部首
    #获得偏旁部首特征，对于没有偏旁部首的字体表标上PAD
    data['radical'] = [[radical.trans_ch(x) if radical.trans_ch(x) is not None else 'UNK' for x in s] for s in texts]
    #获得拼音特征，对于没有拼音的字表标上PAD
    data['pinyin'] = [[pinyin.trans_ch(x) if pinyin.trans_ch(x) is not None else 'UNK' for x in s] for s in texts]

#---------------------------存储数据-------------------------------------------------------
    num_samples = len(texts) #获取多少句话，等于是有多少样本
    num_col = len(data.keys()) #获取特征的个数，也就是列数


    dataset = []
    for i in range(num_samples):
        records = list(zip(*[list(v[i]) for v in data.values()])) #解压
        dataset += records + [['sep'] * num_col] #每存完一个句子需要一行sep进行隔离
    dataset = dataset[:-1] #最后一行sep不要
    dataset = pd.DataFrame(dataset,columns =data.keys()) #转换成dataframe
    save_path = f'data/prepare/{split_name}/{idx}.csv'

    def clean_word(w):
        if w == '\n':
            return 'LB'
        if w in [' ','\t','\u2003']:
            return 'SPACE'
        if w.isdigit(): #将所有的数字都变成一种符号
            return 'num'
        return w
    dataset['word'] = dataset['word'].apply(clean_word)
    dataset.to_csv(save_path,index = False,encoding='utf8')
    # return texts[0],tags[0],bounds[0],flags[0],data['radical'][0],data['pinyin'][0]
def multi_process(split_method = None,train_radio = 0.8):
    if os.path.exists('data/prepare/'):
        shutil.rmtree('data/prepare/')
    if not os.path.exists('data/prepare/train/'):
        os.makedirs('data/prepare/train/')
        os.makedirs('data/prepare/test/')
    idxs = list(set([file.split('.')[0] for file in os.listdir(train_dir)])) #获取所有文件的名字
    shuffle(idxs)#打乱顺序
    index = int(len(idxs) * train_radio) #拿到训练集的截止下标
    train_ids = idxs[:index] #训练集文件名集合
    test_ids = idxs[index:] #测试机文件名集合

    import multiprocessing as mp #并行的方式对数据进行处理
    num_cpus = mp.cpu_count() #获取机器cpu个数
    pool = mp.Pool(num_cpus)
    results = []
    for idx in train_ids:
        result = pool.apply_async(process_text(idx,split_method,'train'))
        results.append(result)
    for idx in test_ids:
        result = pool.apply_async(process_text(idx,split_method,'test'))
        results.append(result)
    pool.close()
    pool.join()
    # [r.get() for r in results]

def mapping(data,threshold = 10,is_word=False,sep = 'sep',is_label = False):
    count = Counter(data)
    if sep is not None:
        count.pop('sep')
    if is_word:
        # ##########
        # #########P
        # ######PPPP
        # ###PPPPPPP

        count['PAD'] = 10000001
        count['UNK'] = 10000000
        data = sorted(count.items(),key = lambda x:x[1],reverse=True)
        data = [ x[0] for x in data if x[1]>=threshold] #去掉频率小于threshold元素
        id2item = data
        item2id = {id2item[i]: i for i in range(len(id2item))}
    elif is_label:
        data = sorted(count.items(),key = lambda x:x[1],reverse=True)
        data = [x[0] for x in data]
        id2item = data
        item2id = {id2item[i]: i for i in range(len(id2item))}
    else:
        count['PAD'] = 10000001
        data = sorted(count.items(),key = lambda x:x[1],reverse=True)
        data = [x[0] for x in data]
        id2item = data
        item2id = {id2item[i]: i for i in range(len(id2item))}
    return id2item,item2id

def get_dict():
    map_dict = {}
    from glob import glob
    all_w = []
    all_bound = []
    all_flag = []
    all_label = []
    all_radical = []
    all_pinyin = []
    for file in glob('/data/prepare/train/*.csv') + glob('data/prepare/test/*.csv'):
        df = pd.read_csv(file,sep = ',')
        all_w += df['word'].tolist()
        all_bound += df['bounds'].tolist()
        all_flag += df['flag'].tolist()
        all_label += df['label'].tolist()
        all_radical += df['radical'].tolist()
        all_pinyin += df['pinyin'].tolist()
    map_dict['word'] = mapping(all_w,threshold=20,is_word=True)
    map_dict['bounds'] = mapping(all_bound)
    map_dict['flag'] = mapping(all_flag)
    map_dict['label'] = mapping(all_label,is_label=True)
    map_dict['radical'] = mapping(all_radical)
    map_dict['pinyin'] = mapping(all_pinyin)

    with open(f'data/prepare/dict.pkl','wb') as f:
        pickle.dump(map_dict,f)



if __name__ == '__main__':
    # print(process_text('0',split_method=split_text),'train')
    # multi_process()
    # print(set([file.split('.')[0] for file in os.listdir(train_dir)]))
    # multi_process(split_text)
    get_dict()
    with open(f'data/prepare/dict.pkl','rb') as f:
        data = pickle.load(f)
    print(data['word'])