import os
import subprocess
from multiprocessing import Process, Pool

os.makedirs('./edf', exist_ok=True)
os.makedirs('./download_log', exist_ok=True)

def download_file(path):
    name = path.split('/')[-1]
    log = "./download_log/"+name.split('.')[0]
    subprocess.call("wget {} -o {} -P ./edf".format(path, log), shell=True)
    # os.system("wget {} -o {} -P ./edf".format(path, log))

if __name__ == '__main__':
    with open ('./download_path.txt', 'r') as f:
        lines = f.readlines()
        pool = Pool()
        for l in lines:
            l = l.strip()
            pool.apply_async(func=download_file, args=(l,))
        
        pool.close()
        pool.join()