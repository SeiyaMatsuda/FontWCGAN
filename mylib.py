import pickle
import random
def pickle_dump(obj,path):
    with open(path,mode="wb") as f:
        pickle.dump(obj,f)
def pickle_load(path):
    with open(path,mode="rb") as f:
        data=pickle.load(f)
        return data
def return_index(label,weight):
    dice = list(range(len(label)))
    # 6の目が出やすいように重みを設定する
    # 歪んだサイコロを1000回振ってサンプルを得る
    if len(weight) == 0:
        samples = random.choices(dice)
    else:
        samples = random.choices(dice, k=1, weights=[1 / w ** 2 for w in weight])
    return samples