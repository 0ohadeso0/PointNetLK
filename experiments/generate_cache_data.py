"""
    Example for training a classifier.

    It is better that before training PointNet-LK,
    train a classifier with same dataset
    so that we can use 'transfer-learning.'
"""

import argparse
import os
import sys
import logging
import numpy
import torch
import torch.utils.data
import torchvision
import time
import pickle as pkl
from tqdm import tqdm
# addpath('../')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
import ptlk

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())


def options(argv=None):
    parser = argparse.ArgumentParser(description='PointNet classifier')

    # required.
    parser.add_argument('-i', '--dataset-path', required=True, type=str,
                        metavar='PATH', help='path to the input dataset') # like '/path/to/ModelNet40'
    parser.add_argument('-c', '--categoryfile', required=True, type=str,
                        metavar='PATH', help='path to the categories to be trained') # eg. './sampledata/modelnet40_half1.txt'

    # settings for input data
    parser.add_argument('--dataset-type', default='modelnet', choices=['modelnet', 'shapenet2'],
                        metavar='DATASET', help='dataset type (default: modelnet)')
    

   

    args = parser.parse_args(argv)
    return args

def main(args):
    # dataset
    trainset, testset = get_datasets(args)
    num_classes = len(trainset.classes)

    # training

    run(args, trainset, testset)


def run(args, trainset, testset):
    name = args.categoryfile.strip().split("/")[-1][:-4]
    
   
    train_file_path = os.path.join(args.dataset_path,name+"_train.pkl")
    test_file_path = os.path.join(args.dataset_path,name+"_test.pkl")
    
    # dataloader
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=1, shuffle=False, num_workers=0)
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=1, shuffle=False, num_workers=0)
    train_list = []
    for data in tqdm(trainloader):
        data[0] = data[0][0]
        train_list.append(data)

    pkl.dump(train_list,open(train_file_path,"wb"))


    test_list = []
    for data in tqdm(testloader):
        data[0] = data[0][0]
        test_list.append(data)

    pkl.dump(test_list,open(test_file_path,"wb"))

class ShapeNet2_transform_coordinate:
    def __init__(self):
        pass
    def __call__(self, mesh):
        return mesh.clone().rot_x()

def get_datasets(args):

    cinfo = None
    if args.categoryfile:
        #categories = numpy.loadtxt(args.categoryfile, dtype=str, delimiter="\n").tolist()
        categories = [line.rstrip('\n') for line in open(args.categoryfile)]
        categories.sort()
        c_to_idx = {categories[i]: i for i in range(len(categories))}
        cinfo = (categories, c_to_idx)

    if args.dataset_type == 'modelnet':
        transform = torchvision.transforms.Compose([\
                ptlk.data.transforms.Mesh2Points(),\
                ptlk.data.transforms.OnUnitCube(),\
                ptlk.data.transforms.Resampler(4096),\
            ])

        trainset = ptlk.data.datasets.ModelNet(args.dataset_path, train=1, transform=transform, classinfo=cinfo)
        testset = ptlk.data.datasets.ModelNet(args.dataset_path, train=0, transform=transform, classinfo=cinfo)

    elif args.dataset_type == 'shapenet2':
        transform = torchvision.transforms.Compose([\
                ShapeNet2_transform_coordinate(),\
                ptlk.data.transforms.Mesh2Points(),\
                ptlk.data.transforms.OnUnitCube(),\
                ptlk.data.transforms.Resampler(4096),\
            ])

        dataset = ptlk.data.datasets.ShapeNet2(args.dataset_path, transform=transform, classinfo=cinfo)
        trainset, testset = dataset.split(0.8)

    return trainset, testset


if __name__ == '__main__':
    ARGS = options()


    main(ARGS)
    LOGGER.debug('done (PID=%d)', os.getpid())

#EOF