import os
import scipy.io
import numpy as np

root = 'data/ND-code-datasets'
dir_list = ['Application1-gene-regulatory-network/networks', 
            'Application2-protein-contact-maps/full-predictions', 
            'Application3-coauthorship-network/data']

for _dir in dir_list:
    dir = os.path.join(root, _dir)
    files = os.listdir(dir)
    for cnt, filename in enumerate(files):
        print(f"{cnt + 1} / {len(files)}")
        if filename.endswith('.mat'):
            file_path = os.path.join(dir, filename)
            absolute_path = os.path.abspath(file_path)
            new_file_path = os.path.splitext(absolute_path)[0] + '.npy'
            if os.path.exists(new_file_path):
                continue
            try:
                mat = scipy.io.loadmat(absolute_path)
                keys = list(mat.keys())
                assert keys[:3] == ['__header__', '__version__', '__globals__'], \
                    f"Error: {keys[:3]} is wrong"
                assert len(mat.keys()) == 4, \
                    f"Error: {absolute_path} network number is wrong"
                mat = mat[list(mat.keys())[3]]
                np.save(new_file_path, mat)
            except Exception as e:
                print(e)

print("Done")
            