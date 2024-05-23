comp2id = {}
with open("raw/vocab_compound") as f:
    for index, line in enumerate(f):
        comp2id[line.strip()] = index
print(comp2id)

import torch

def insert_padding(mylist, max_size):
    for i, sentence in enumerate(mylist):
        templist = [0 for _ in range(max_size-len(sentence))]
        mylist[i] = mylist[i] + templist
    return mylist

def process_independent(mylist, max_size, file_name):
    mylist = insert_padding(mylist, max_size)
    mytensor = torch.tensor(mylist)
    torch.save(mytensor, 'processed/'+file_name+".pt")
    return mytensor

comp_MAX_size = 100

from tqdm import tqdm
import numpy as np
import gc

for file_num in range(10):  # 0부터 9까지 파일 번호
    file_path = f'intermediate/chunk_{file_num}.tsv'
    smiles_ids = []  # 현재 파일의 결과를 저장할 리스트

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in tqdm(f, desc=f"Processing chunk {file_num}"):
            try:
                sentence = line.strip().split('\t')[0]  # SMILES 문자열만 추출
                temp = []
                skip_string = False
                for char in sentence:
                    if char not in comp2id:
                        skip_string = True
                        break
                    else:
                        temp.append(comp2id[char])
                if not skip_string and len(temp) <= comp_MAX_size:  # 길이 조건 추가
                    smiles_ids.append(temp)
            except UnicodeDecodeError:
                continue  # UnicodeDecodeError가 발생하면 이 라인을 건너뛰고 계속 진행

    process_independent(smiles_ids, comp_MAX_size, f"smiles_ids_{file_num}")

    # 파일 처리 후 메모리 관리
    smiles_ids = []
    gc.collect()  # 파일 하나 처리 후 강제 가비지 컬렉션