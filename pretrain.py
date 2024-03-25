# set cwd set to matrixssl-inductive
import os
os.chdir(os.path.dirname(__file__))
from data.loader import generate_cube_data

from torch.utils.data import TensorDataset, DataLoader

# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("task", help="specifies synthetic data generating process, one of 'cube_single', ")
#     parser.add_argument("backbone", help="architecture used, one of 'linear', ")
#     parser.add_argument("-n", help="number of datum", type=int)
#     parser.add_argument("-d", help="dimensionality of data", type=int)
#     parser.add_argument("-k", help="number invariant dimensions", type=int)
#     parser.add_argument("--split", help="specifies train-val split; input some number < n", type=int)
#     # might need conditional arguments... ?
#     args = parser.parse_args()
#     return args

# ideally: just provide args for model, then train
def main():
    (x1, x2, _), (val_x, val_y) = generate_cube_data()
    epochs = 100

    # create dataloader ?
    trainset, valset = TensorDataset(x1, x2), TensorDataset(val_x, val_y)
    trainloader = DataLoader(trainset, batch_size=32)
    valloader = DataLoader(valset, batch_size=32)

    model = None
    model.train_model()



    pass


if __name__ == '__main__':
    main()
