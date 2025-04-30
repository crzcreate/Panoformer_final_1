import os
txt_dir="E:\HorizonNet-master\data\layoutnet_dataset\\test\label_cor"
num_list=[]
for i in os.listdir(txt_dir):
    with open(os.path.join(txt_dir,i), 'r', encoding='utf-8') as file:
        lines = file.readlines()
    # 计算行数
    num_lines = len(lines)
    if num_lines!=8:
        print(f"not curbid{i}")