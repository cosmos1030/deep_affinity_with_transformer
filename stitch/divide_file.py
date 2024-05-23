from tqdm import tqdm

smiles = []
with open("raw/chemicals.v5.0.tsv", 'r+', encoding="ISO-8859-1") as f:
    for line in f:
        smiles.append(line.strip().split('\t')[3])

smiles = smiles[1:]


def save_as_tsv(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for row in data:
            f.write('\t'.join(map(str, row)) + '\n')

def split_and_save_list(big_list, chunk_size):
    total = len(big_list) // chunk_size + (1 if len(big_list) % chunk_size else 0)  # 전체 청크의 개수 계산
    for i in range(0, len(big_list), chunk_size, total=total):
        chunk = big_list[i:i + chunk_size]
        filename = f'intermediate/chunk_{i//chunk_size}.tsv'
        save_as_tsv(chunk, filename)
        chunk = None

split_and_save_list(smiles, len(smiles)//50)