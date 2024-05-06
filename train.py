import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import r2_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet101
import torchvision.models as models
import transforms as trans

import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
import time

batchsize = 10 # 批大小,
oct_img_size = [512, 512]
image_size = 256
iters = 1000# 迭代次数
val_ratio = 0.2 # 训练/验证数据划分比例，80 / 20
trainset_root = "../STAGE_training/training_images"
test_root = "../STAGE_validation/validation_images"
num_workers = 8
64 = 1e-4
optimizer_type = "adam"
labellist = {'normal':0,'early':1,'intermediate':2,'advanced':3}

filelists = os.listdir(trainset_root)
train_filelists, val_filelists = train_test_split(filelists, test_size=val_ratio, random_state=42)
print("Total Nums: {}, train: {}, val: {}".format(len(filelists), len(train_filelists), len(val_filelists)))


class STAGE_dataset(torch.utils.data.Dataset):
    """
    getitem() output:
    
    	fundus_img: RGB uint8 image with shape (3, image_size, image_size)
        
        oct_img:    Uint8 image with shape (256, oct_img_size[0], oct_img_size[1])
    """

    def __init__(self,
                 transforms,
                 dataset_root,
                 label_file='',
                 label_file2='',
                 filelists=None,
                 num_classes=3,
                 mode='train'):

        self.dataset_root = dataset_root
        self.oct_transforms = transforms
        self.mode = mode.lower()
        self.num_classes = num_classes
        
        if self.mode == 'train':
            label = {row['ID']: row[1]
                        for _, row in pd.read_excel(label_file).iterrows()}
            label2 = {row['ID']: row[4]
                        for _, row in pd.read_excel(label_file2).iterrows()}
            self.file_list = [[f, label[int(f)], label2[int(f)]] for f in os.listdir(dataset_root)]
        elif self.mode == "test":
            label2 = {row['ID']: row[4]
                        for _, row in pd.read_excel(label_file2).iterrows()}
            self.file_list = [[f, label2[int(f)],None] for f in os.listdir(dataset_root)]
        
        if filelists is not None:
            self.file_list = [item for item in self.file_list if item[0] in filelists]

    def __getitem__(self, idx):
        real_index, label , label2 = self.file_list[idx]

        oct_series_list = sorted(os.listdir(os.path.join(self.dataset_root, real_index)), 
                                    key=lambda x: int(x.strip("_")[0]))

        oct_series_0 = cv2.imread(os.path.join(self.dataset_root, real_index, oct_series_list[0]), 
                                    cv2.IMREAD_GRAYSCALE)
        oct_img = np.zeros((len(oct_series_list), oct_series_0.shape[0], oct_series_0.shape[1], 1), dtype="uint8")

        for k, p in enumerate(oct_series_list):
            oct_img[k] = cv2.imread(
                os.path.join(self.dataset_root, real_index, p), cv2.IMREAD_GRAYSCALE)[..., np.newaxis]

        if self.oct_transforms is not None:
            oct_img = self.oct_transforms(oct_img)
            oct_img = np.ascontiguousarray(oct_img)

 
        # normlize on GPU to save CPU Memory and IO consuming.
        # fundus_img = (fundus_img / 255.).astype("float32")
        # oct_img = (oct_img / 255.).astype("float32")

        oct_img = oct_img.squeeze(-1) # D, H, W, 1 -> D, H, W

        if self.mode == 'test':
            return oct_img, real_index ,labellist[label]
        if self.mode == "train":
            return oct_img, label, labellist[label2]

    def __len__(self):
        return len(self.file_list)


class Model(nn.Module):
    """
    simply create a 2-branch network, and concat global pooled feature vector.
    each branch = single resnet34
    """
    def __init__(self):
        super(Model, self).__init__()
        # self.branch = resnet101(pretrained=True, num_classes=0) # 移除最后一层全连接层
        self.branch = models.resnext101_64x4d(pretrained=True)
        self.branch.fc = nn.Linear(in_features=2048,out_features=4)
        self.linear2 = nn.Linear(4, 1) 
        self.sigmoid = nn.Sigmoid()
        # 在oct_branch更改第一个卷积层通道数
        self.branch.conv1 = nn.Conv2d(256, 64,
                                        kernel_size=7,
                                        stride=2,
                                        padding=3)
        


    def forward(self,input):
        blogit = self.branch(input)

        logit = self.linear2(blogit)
        logit = self.sigmoid(logit)
        return logit , blogit



class SMAPELoss(nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, labels,preds,):
        return 1 / len(preds) * torch.sum(2 * torch.abs(preds - labels) / (torch.abs(preds) + torch.abs(labels)))

def Smape_(labels,preds):
    return 1 / len(preds) * torch.sum(2 * torch.abs(preds - labels) / (torch.abs(preds) + torch.abs(labels)))
               
def process(logits,labels):
    logits = logits.squeeze()
    label = [ 1 if i == 0 else i * (-6)  for i in labels.numpy()] 
    return logits * torch.Tensor(label)
  
def R2(preds, labels):
    return 1 - torch.sum((preds - labels) ** 2) / torch.sum((labels - labels.mean()) ** 2)

def Score(smape, R2):
    return  0.5 * (1 / (smape + 0.1)) + 0.5 * (R2 * 10)


def train(model,iters, train_dataloader, val_dataloader, optimizer, criterion, log_interval, eval_interval):
    iter = 0
    model.train()
    avg_loss_list = []
    avg_score_list = []
    avg_kappa_list = []
    best_score = -999
    t = time.time()
    while iter <iters:
        for data in train_dataloader:
            iter += 1
            if iter > iters:
                break

            oct_imgs = (data[0] / 255.).to(torch.float32)
            labels = data[1].to(torch.float32)
            labels2 = data[2].to(torch.int64)

            logits , blogits = model(oct_imgs.cuda())
            logits = logits.cpu()
            blogits = blogits.cpu()
            logits = process(logits, labels2)
            

            smape = Smape_(labels,logits)
            r2 = r2_score(labels.detach().numpy(),logits.detach().numpy())
            loss = criterion(labels,logits)

            
            avg_score_list.append(Score(smape.detach().numpy(),r2))
            # loss = loss.detach().numpy()
            # print(Score(smape.detach().numpy(),r2),loss)
            avg_loss_list.append(loss.detach().numpy())
            
            for p, l in zip(blogits.detach().numpy().argmax(1), labels2.detach().numpy()):
                avg_kappa_list.append([p, l])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()



            
            if iter % log_interval == 0:
                avg_loss = np.array(avg_loss_list).mean()
                avg_score = np.array(avg_score_list).mean()
                avg_loss_list = []
                avg_score_list = []
                avg_kappa_list = np.array(avg_kappa_list)
                avg_kappa = cohen_kappa_score(avg_kappa_list[:, 0], avg_kappa_list[:, 1], weights='quadratic')
                avg_kappa_list = []
                
                print("[TRAIN] iter={}/{} avg_loss={:.4f} avg_acore={:.4f} avg_kappa={:.4f}".format(iter, iters, avg_loss, avg_score,avg_kappa),time.time()-t)
                t = time.time()

            if iter % eval_interval == 0:
                avg_loss, avg_score, avg_kappa = val(model, val_dataloader, criterion)
                print("[EVAL] iter={}/{} avg_loss={:.4f} acore={:.4f} akappa={:.4f}".format(iter, iters, avg_loss, avg_score,avg_kappa))
                if avg_score >= best_score:
                    best_score = avg_score
                    torch.save(model.state_dict(),
                            os.path.join("best_model_{:.4f}".format(best_score), 'model.pt'))
                model.train()

def val(model, val_dataloader, criterion):
    model.eval()
    avg_loss_list = []
    avg_kappa_list = []
    avg_score_list = []
    with torch.no_grad():
        for data in val_dataloader:
            oct_imgs = (data[0] / 255.).to(torch.float32)
            labels = data[1].to(torch.float32)
            labels2 = data[2].to(torch.int64)
            logits , blogits = model(oct_imgs.cuda())
            logits = logits.cpu()
            blogits = blogits.cpu()
            
            logits = process(logits, labels2)
            smape = Smape_(labels,logits)
            r2 = r2_score(labels.detach().numpy(),logits.detach().numpy())
            
            for p, l in zip(blogits.numpy().argmax(1), labels2.detach().numpy()):
                avg_kappa_list.append([p, l])
                
            loss = criterion(labels,logits)

            avg_loss_list.append(loss.detach().numpy())
            avg_score_list.append(Score(smape.detach().numpy(),r2))
            
    avg_score = np.array(avg_score_list).mean()
    avg_loss = np.array(avg_loss_list).mean()
    avg_kappa_list = np.array(avg_kappa_list)
    avg_kappa = cohen_kappa_score(avg_kappa_list[:, 0], avg_kappa_list[:, 1], weights='quadratic')
    return avg_loss, avg_score , avg_kappa

train_transforms = trans.Compose([
    trans.CenterCrop([256] + oct_img_size),
    trans.RandomHorizontalFlip(),
    trans.RandomVerticalFlip()
])

val_transforms = trans.Compose([
    trans.CenterCrop([256] + oct_img_size)
])

train_dataset = STAGE_dataset(dataset_root=trainset_root, 
                        transforms=train_transforms,
                        filelists=train_filelists,
                        label_file='../STAGE_training/training_GT/task1_GT_training.xlsx',
                        label_file2='../STAGE_training/data_info_training.xlsx')

val_dataset = STAGE_dataset(dataset_root=trainset_root, 
                        transforms=val_transforms,
                        filelists=val_filelists,
                        label_file='../STAGE_training/training_GT/task1_GT_training.xlsx',
                        label_file2='../STAGE_training/data_info_training.xlsx')


train_loader = torch.utils.data.DataLoader(
    train_dataset,

    batch_size=batchsize, 
    shuffle=True,
    num_workers=num_workers,

)

val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=batchsize, 
    shuffle=True,
    num_workers=num_workers,

)


model = Model().cuda()

if optimizer_type == "adam":
    optimizer = torch.optim.Adam(model.parameters(),init_lr)

criterion = SMAPELoss()


model.train()
train(model, iters, train_loader, val_loader, optimizer, criterion, log_interval=10, eval_interval=100)

oct_test_transforms = trans.Compose([
    trans.CenterCrop([256] + oct_img_size)
])

test_dataset = STAGE_dataset(dataset_root=test_root, 
                        transforms=oct_test_transforms,
                        label_file2='../STAGE_validation/data_info_validation.xlsx',
                        mode='test')
cache = []
for oct_img, idx , labels2 in test_dataset:
    oct_img = oct_img[np.newaxis, ...]
    oct_img = torch.Tensor(oct_img / 255.).to(torch.float32)

    logits , blogits = model(oct_img.cuda())
    logits = logits.cpu()
    blogits = blogits.cpu()
    if labels2 != 0:
        logits = logits.squeeze() * labels2 * (-6)

    cache.append([idx, logits.detach().numpy()])
submission_result = pd.DataFrame(cache, columns=['ID', 'pred_MD'])
submission_result.to_csv("./MD_Results.csv", index=False)

