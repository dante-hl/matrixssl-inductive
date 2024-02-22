import argparse  # ok to do here?
# set cwd set to matrixssl-inductive
import os
os.chdir(os.path.dirname(__file__))
from data import loader

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("task", help="specifies synthetic data generating process, one of 'cube_single', ")
    parser.add_argument("backbone", help="architecture used, one of 'linear', ")
    parser.add_argument("-n", help="number of datum", type=int)
    parser.add_argument("-d", help="dimensionality of data", type=int)
    parser.add_argument("-k", help="number invariant dimensions", type=int)
    parser.add_argument("--split", help="specifies train-test split; input some number < n", type=int)
    # might need conditional arguments... ?
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    n, d, k = args.n, args.d, args.k
    # generate data
    if args.task == 'cube_single':
        full_data = loader.generate_cube_data(n, d, k, labelling='single')
        train_data = full_data[]


    pass


if __name__ == '__main__':
    main()
