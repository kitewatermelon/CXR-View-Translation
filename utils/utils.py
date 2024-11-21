import os 
from datetime import datetime

def make_dirs(mode):
    i = 0
    save_path = rf"results/{mode + datetime.today().strftime('%y%m%d')}/{i}/"
    while os.path.exists(save_path):
        save_path = rf"results/{mode + datetime.today().strftime('%y%m%d')}/{i}/"
        i += 1

    paths = ['img/train/', 'img/test/', 'history/', 'metrics/','model/']
    for path in paths:
        os.makedirs(save_path + path, exist_ok=True)
    return save_path

if __name__ == "__main__":
    make_dirs('L2P')