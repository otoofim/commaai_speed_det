import torchvision.models as models
from torchvision import transforms
from dataloader import *
import numpy as np
import torch
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torchvision
import wandb
from arch import *
import os
import warnings
warnings.filterwarnings("ignore")





class Resnet50(object):

    def __init__(self):
        self.resnet50 = wide_resnet50_2(pretrained=True).eval()
        self.device = None
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        self.resnet50 = self.resnet50.to(self.device)

    def __call__(self, sample):
        sample = sample.to(self.device)
        with torch.no_grad():
            out = self.resnet50(sample.unsqueeze(0))
            return out.squeeze()


def train(batch_size, epoch, learning_rate, run_name, data_path, project_name, arch_name, dataset_name, continue_tra = False, model_add = None, wandb_id = None):


    hyperparameter_defaults = {
        "batch_size": batch_size,
        "lr": learning_rate,
        "epochs": epoch,
        "momentum": 0.9,
        "architecture": arch_name,
        "dataset": dataset_name,
        "run": run_name
    }

    base_add = os.getcwd()


    if continue_tra:
        wandb.init(config = hyperparameter_defaults, project = project_name, entity = 'moh1371',
                    name = hyperparameter_defaults['run'], resume = "must", id = wandb_id)
        print("wandb resumed...")
    else:
        wandb.init(config = hyperparameter_defaults, project = project_name, entity = 'moh1371',
                    name = hyperparameter_defaults['run'], resume = "allow")


    val_every = 5
    img_w = 256
    img_h = 256


    preprocess_in = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Resize((img_h, img_w)),
        #Resnet50()
    ])


    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Running on the GPU")
    else:
        device = torch.device("cpu")
        print("Running on the CPU")


    model = half_UNet((img_h, img_w), out_channels = 3)
    #model = MyConv(img_w)
    #model = nvidia(3, (img_w, img_h)).apply(nvidia.init_weights)
    #model = mynvidia(3, (img_w, img_h))
    #model = NVIDIA(3)
    if continue_tra:
        model.load_state_dict(torch.load(model_add)['model_state_dict'])
        print("model state dict loaded...")

    model = model.to(device)

    #tr_loader = commaai(data_path, preprocess_in)
    tr_loader = winterdata("../signals/data/signal dataset", preprocess_in)
    tra, val = random_split(tr_loader, [int(len(tr_loader) * 0.8), int(len(tr_loader) * 0.2)])
    train_loader = DataLoader(dataset = tra, batch_size = wandb.config.batch_size, shuffle = False)
    val_loader = DataLoader(dataset = val, batch_size = wandb.config.batch_size, shuffle = False)


    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = wandb.config.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=0, threshold=0.2, threshold_mode='abs', min_lr=1e-8)


    # optimizer = optim.SGD(model.parameters(), lr = wandb.config.lr, momentum = wandb.config.momentum)
    # if continue_tra:
    #     optimizer.load_state_dict(torch.load(model_add)['optimizer_state_dict'])
    #     print("optimizer state dict loaded...")
    #
    # criterion = torch.nn.MSELoss(reduction='mean')


    tr_loss = 0.0
    val_loss = 0.0
    best_val = 1e10
    wandb.watch(model)

    start_epoch = 0
    end_epoch = wandb.config.epochs

    if continue_tra:
        start_epoch = torch.load(model_add)['epoch'] + 1
        end_epoch = torch.load(model_add)['epoch'] + 1 + int(wandb.config.epochs)


    with tqdm(range(start_epoch, end_epoch), unit="epoch", leave = True, position = 0) as epobar:
        for epoch in epobar:
                epobar.set_description("Epoch {}".format(epoch + 1))
                epobar.set_postfix(ordered_dict = {'tr_loss':tr_loss, 'val_loss':val_loss})

                tr_loss = 0.0
                out = None
                images = None
                labels = None

                with tqdm(train_loader, unit="batch", leave = False) as batbar:
                    for i, batch in enumerate(batbar):


                        batbar.set_description("Batch {}".format(i + 1))

                        optimizer.zero_grad()

                        out = model.train()(batch['image'].to(device))
                        #out = model.train()(batch['optical'].to(device), batch['image'].to(device))
                        loss = criterion(out.float(), batch['label'].float().to(device))
                        loss.backward()

                        optimizer.step()
                        #print(float(loss.item()))
                        tr_loss += float(loss.item())

                        images = batch['image']
                        labels = batch['label']


                #org_img = {'input':wandb.Image(batch['image']),
                #"ground truth":wandb.Image(batch['label']),
                #"prediction":wandb.Image(out)}

                #wandb.log(org_img)


                tr_loss /= len(train_loader)
                wandb.log({"tr_loss": tr_loss, "epoch": epoch + 1})


                if ((epoch+1) % val_every == 0):
                    with tqdm(val_loader, unit="batch", leave = False) as valbar:
                        with torch.no_grad():
                            val_loss = 0.0
                            for i, batch in enumerate(valbar):

                                valbar.set_description("Val_batch {}".format(i + 1))

                                out = model.eval()(batch['image'].to(device))
                                #out = model.eval()(batch['optical'].to(device), batch['image'].to(device))

                                loss = criterion(out, batch['label'].to(device))

                                val_loss += float(loss.item())


                        val_loss /= len(val_loader)
                        scheduler.step(val_loss)
                        wandb.log({"val_loss": val_loss, "epoch": epoch + 1})
                        if val_loss < best_val:

                            newpath = base_add + "/checkpoints/{}".format(hyperparameter_defaults['run'])

                            if not os.path.exists(base_add + "/checkpoints"):
                                os.makedirs(base_add + "/checkpoints")

                            if not os.path.exists(newpath):
                                os.makedirs(newpath)

                            torch.save({
                                'epoch': epoch,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'tr_loss': tr_loss,
                                'val_loss': val_loss,
                                }, newpath + "/best.pth")
