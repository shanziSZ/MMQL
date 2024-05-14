import argparse
import torch
from torch.utils.data import DataLoader

from dataset_RAD_group import RADGroupFeatureDataset
from dataset_SLAKE_group import SLAKEGroupFeatureDataset
from dataset_SLAKE_star_group import SLAKEStarGroupFeatureDataset
from model_grouped import model
from train_grouped import train, test


def parse_args():
    parser = argparse.ArgumentParser(description="Med VQA")

    # GPU config
    parser.add_argument('--seed', type=int, default=717, help='random seed for gpu.default:5')
    parser.add_argument('--gpu', type=int, default=0, help='use gpu device. default:0')

    # Model loading/saving
    parser.add_argument('--input', type=str, default=None, help='input file directory for continue training from stop one')
    parser.add_argument('--output', type=str, default='saved_models', help='save file directory')

    # Training testing or sampling Hyper-parameters
    parser.add_argument('--epochs', type=int, default=200, help='the number of epoches')
    # parser.add_argument('--lr', default=1e-5, type=float, metavar='lr', help='initial learning rate')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--update_freq', default='1', metavar='N', help='update parameters every n batches in an epoch')
    parser.add_argument('--print_interval', default=20, type=int, metavar='N', help='print per certain number of steps')

    # Train
    parser.add_argument('--use_data', action='store_true', default=True, help='Using TDIUC dataset to train')
    parser.add_argument('--data_dir', type=str, help='RAD dir')

    # Details
    parser.add_argument('--details', type=str, default='original ')

    # Dataset name
    parser.add_argument('--dataset_name', type=str, default='SLAKE')

    args = parser.parse_args()

    return args


if __name__ == '__main__':

    root = '.'
    args = parse_args()
    if args.dataset_name == "SLAKE":
        data = root + '/data-SLAKE'
    else:
        data = root + '/data-RAD'
    args.data_dir = data

    # Set GPU device
    device = torch.device("cuda:" + str(args.gpu) if args.gpu >= 0 else "cpu")
    args.device = device

    # Fixed ramdom seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    # Prepare the dataloader
    if args.dataset_name == "SLAKE":
        train_dataset = SLAKEGroupFeatureDataset('train', args, dataroot=data)
        val_dataset = SLAKEGroupFeatureDataset('val', args, dataroot=data)
        test_dataset = SLAKEGroupFeatureDataset('test', args, dataroot=data)
        # train_dataset = SLAKEStarGroupFeatureDataset('train', args, dataroot=data)
        # test_dataset = SLAKEStarGroupFeatureDataset('test', args, dataroot=data)
    else:
        train_dataset = RADGroupFeatureDataset('train', args, dataroot=data)
        test_dataset = RADGroupFeatureDataset('test', args, dataroot=data)
    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=0, drop_last=False, pin_memory=True)
    if args.dataset_name == "SLAKE":
        val_loader = DataLoader(val_dataset, 16, shuffle=False, num_workers=0, drop_last=False,pin_memory=True)
    test_loader = DataLoader(test_dataset, 16, shuffle=False, num_workers=0, drop_last=False, pin_memory=True)


    # Build model
    model_instance = model(args)

    # Load snapshot
    ckpt = './saved_models/2024Jan26-094011_rebuttal_SLAKE_0.3_provided/74_model.pth'
    if ckpt is not None:
        print('loading %s' % ckpt)
        pre_ckpt = torch.load(ckpt)
        print(pre_ckpt.keys())
        model_instance.load_state_dict(pre_ckpt.get('model_state', pre_ckpt))
        epoch = pre_ckpt['epoch'] + 1

    # train(args, model_instance, train_loader, val_loader)
    test(args, model_instance, test_loader)