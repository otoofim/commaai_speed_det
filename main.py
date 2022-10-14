import argparse
import os
import sys
from train import *



def main():

    parser = argparse.ArgumentParser()

    #512
    parser.add_argument('--batch_size', '-bs', default = 128, type = int,
                        help = 'mini batches size')


    parser.add_argument('--epoch', '-e', default = 100, type = int,
                        help = 'num of epochs')


    parser.add_argument('--learning_rate', '-lr', default = .0001, type = float,
                        help = 'learning rate')

    #**********
    parser.add_argument('--data_path', '-dp',
                        default = "./data", type = str,
                        help = 'path to commaai dataset. It should be like PATH/dataset')

    parser.add_argument('--run_name', '-rn',
                        default = "commaai", type = str,
                        help = 'the run name will be apeared in wandb')

    parser.add_argument('--project_name', '-pn',
                        default = "speed detection", type = str,
                        help = 'the run name will be apeared in wandb')

    parser.add_argument('--continue_tra', '-ct',
                        default = False, type = bool,
                        help = 'train a model for more epochs. you also need to set the model path.')

    parser.add_argument('--arch_name', '-an',
                        default = "unet", type = str,
                        help = 'Name of architecture')

    parser.add_argument('--dataset_name', '-dn',
                        default = "winter", type = str,
                        help = 'Name of dataset')

    parser.add_argument('--model_path', '-mp',
                        default = "./checkpoints/test/best.pth", type = str,
                        help = 'The path to the model is going to be trained.')
    #***********
    parser.add_argument('--wandb_id', '-id',
                        default = "test", type = str,
                        help = 'Corresponding wandb run id to resume training.')


    args = parser.parse_args()

    args = {"batch_size":args.batch_size, "epoch":args.epoch, "learning_rate":args.learning_rate, "run_name":args.run_name, "data_path":args.data_path, "project_name":args.project_name, "arch_name":args.arch_name, "dataset_name":args.dataset_name, "continue_tra":args.continue_tra, "model_path":args.model_path, "wandb_id":args.wandb_id}


    #train(args.batch_size, args.epoch, args.learning_rate, args.run_name, args.data_path, args.project_name, args.arch_name, args.dataset_name, args.continue_tra, args.model_path, args.wandb_id,)
    train(args)



if __name__ == "__main__":
    main()
