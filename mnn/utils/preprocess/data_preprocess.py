# -*- coding: utf-8 -*-
import scipy.io
import os
import shutil


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def imagenet_val_preprocess(val_dir='./val', devkit_dir='./ILSVRC2012_devkit_t12'):
    synset = scipy.io.loadmat(os.path.join(devkit_dir, 'data', 'meta.mat'))
    with open(os.path.join(devkit_dir, 'data', 'ILSVRC2012_validation_ground_truth.txt'), 'r') as f:
        lines = f.readlines()
        labels = [int(line[:-1]) for line in lines]
    root, _, filenames = next(os.walk(val_dir))
    for filename in filenames:
        val_id = int(filename.split('.')[0].split('_')[-1])
        ILSVRC_ID = labels[val_id - 1]
        WIND = synset['synsets'][ILSVRC_ID - 1][0][1][0]
        print("val_id:%d, ILSVRC_ID:%d, WIND:%s" % (val_id, ILSVRC_ID, WIND))

        output_dir = os.path.join(root, WIND)
        if os.path.isdir(output_dir):
            pass
        else:
            os.mkdir(output_dir)
        shutil.move(os.path.join(root, filename), os.path.join(output_dir, filename))
