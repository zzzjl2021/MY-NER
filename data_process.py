import os
import re
from typing import Text

train_dir = './ruijin_round1_train2_20181022'
def get_entities(dir):
    """
    function:get the entities' type
    param: the file's path
    return:entities' type
    """
    entities = {} # store entity
    files = os.listdir(dir)
    # print(type(files))
    files = list(set([file.split('.')[0] for file in files])) 
    for file in files:
        path=os.path.join(dir,file+'.ann')
        with open(path,'r',encoding='utf8') as f:
            for line in f.readlines():
                # print(line.split('\t')[1].split(' ')[0])
                name = line.split('\t')[1].split(' ')[0]
                if name in entities:
                    entities[name]+=1
                else:
                    entities[name]=1
    return entities

def get_labelencoder(entities):
    """
    function: get the labels and the id of labels
    :param:entities
    :return: labels,label2id
    """
    entities = sorted(entities.items(),key=lambda x:x[1],reverse = True)
    print(entities)
    entities = [x[0] for x in entities]
    label = []
    label.append('O')
    for entity in entities:
        label.append('B-'+ entity)
        label.append('I-'+ entity)
    label2id = {label[i]:i for i in range(len(label))}
    return label,label2id



def split_text(text):
    split_index=[]
    pattern1 = '\([一二三四五六七八九十]\)|[一二三四五六七八九十]、'
    pattern1 += '【|[\d、|[A-Z]、|\d\.'
    for m in re.finditer(pattern1, text):
        idx=m.span()[0]
        if text[idx-1].isdigit() and text[idx+1].isdigit():
            continue
        split_index.append(idx)
    split_index = list(set([0, len(text)]+split_index))
    pattern2 = '。|；|，|,'
    for m in re.finditer(pattern2, text):
        idx=m.span()[0]
        split_index.append(idx+1)

    split_index = list(sorted(set([0, len(text)] + split_index)))
    # 长短句处理
    other_index=[]
    for i in range(len(split_index)-1):
        begin = split_index[i]
        end = split_index[i+1]
        if text[begin] in '一二三四五六七八九十' or (text[begin]=='(' and text[begin+1] in '一二三四五六七八九十'):
            for j in range(begin, end):
                if text[j] == '\n':
                    other_index.append(j+1)
    split_index+=other_index
    split_index = list(sorted(set([0, len(text)] + split_index)))

    other_index=[]
    for i in range(len(split_index)-1):
        b = split_index[i]
        e = split_index[i+1]
        other_index.append(b)
        if e-b>150:
            for j in range(b, e):
                if (j+1-other_index[-1]) > 15:  # 保证句长>15
                    if text[j] == '\n':
                        other_index.append(j+1)
                    if text[j]==' ' and text[j-1].isnumeric() and text[j+1].isnumeric():
                        other_index.append(j+1)
    split_index += other_index
    split_index = list(sorted(set([0, len(text)] + split_index)))

    # 处理短句子  拼接
    for i in range(1, len(split_index)-1):  # 干掉全部空格的句子
        idx=split_index[i]
        while idx>split_index[i-1]-1 and text[idx-1].isspace():
            idx-=1
        split_index[i] = idx
    split_index = list(sorted(set([0, len(text)] + split_index)))

    # 处理短句子
    temp_idx=[]
    i=0
    while i < len(split_index)-1:
        b=split_index[i]
        e=split_index[i+1]

        num_ch=0
        num_en=0
        if e-b<15:
            for ch in text[b:e]:
                if ischinese(ch):
                    num_ch+=1
                elif ch.islower() or ch.isupper():
                    num_en += 1
                if num_ch+0.5*num_en>5:
                    temp_idx.append(b)
                    i+=1
                    break
            if num_ch+0.5*num_en<=5:
                temp_idx.append(b)
                i+=2
        else:
            temp_idx.append(b)
            i+=1
    split_index = list(sorted(set([0, len(text)] + temp_idx)))
    result=[]
    for i in range(len(split_index)-1):
        result.append(text[split_index[i]:split_index[i+1]])

    # 做检查
    s=''
    for r in result:
        s+=r
    assert len(s)==len(text)
    return result

def ischinese(char):
    if '\u4e00' <= char <= '\u9fff':
        return True
    return False



if __name__ == '__main__':
    # entities = get_entities(train_dir)
    # label = get_labelencoder(entities)
    # pattern = '。|，|,|;|；|\.'
    # with open('./my_ner/ruijin_round1_train2_20181022/0.txt','r',encoding='utf8') as f:
    #     text=f.read()
    #     print(text)
    #     for m in re.finditer(pattern,text):
    #         # print(m)
    #         start = m.span()[0]-5
    #         end = m.span()[1]+5
    #         print('****',text[start:end],'****')
    #         print(text[start+5])
    files = os.listdir(train_dir)
    files = list(set([file.split('.')[0] for file in files]))

    # for file in files:
    #     path = os.path.join(train_dir,file+'.txt')
    #     with open(path,'r',encoding='utf8') as f:
    #         text = f.read()
    #         pattern1 = '。|，|,|;|；|\.|\？'
    #         for m in re.finditer(pattern1,text):
    #             idx = m.span()[0]
    #             if text[idx-1] == '\n':
    #                 print(file,text[idx-10:idx+10])
    
    path = os.path.join(train_dir,files[1]+'.txt')
    with open(path,'r',encoding='utf8') as f:
        text = f.read()
        a=split_text(text)
   