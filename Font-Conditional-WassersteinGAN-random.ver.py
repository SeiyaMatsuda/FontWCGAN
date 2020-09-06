import torch
import torch.nn as nn
import os
from torchvision.utils import save_image
import numpy as np
import glob
import time
import torch.autograd as autograd
import random
import tqdm
import datetime
from mylib import *


class Transform(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        return (sample/127.5)-1


transform = Transform()
sortsecond=lambda a : os.path.splitext(os.path.basename(a))[0]
Font_files =sorted(glob.glob("./Myfont/dataset/img_vector/*.pt"),key=sortsecond)
Label_files=sorted(glob.glob("./Myfont/dataset/tag_vector/*.pt"),key=sortsecond)
tag_prob=pickle_load("./Myfont/dataset/tag_prob/tag_prob.pickle")
p=list(zip(Font_files,Label_files,tag_prob))
random.shuffle(p)
Font_files,Label_files,tag_prob=zip(*p)
#テストデータと訓練データを分ける(MY Fontのフォント数は18815)
ratio=int(len(Font_files)*0.8)
train_Font= Font_files[:ratio]
train_Label= Label_files[:ratio]
test_Font=Font_files[ratio:]
test_Label=Label_files[ratio:]
dt_now=str(datetime.datetime.now())
os.makedirs(os.path.join("./result",dt_now))
log_dir=os.path.join("./result",dt_now)
os.makedirs(os.path.join(log_dir,"logs_cWGAN"))
os.makedirs(os.path.join(log_dir,"checkpoint_cWGAN"))
class dataset_full(torch.utils.data.Dataset):

    def __init__(self, Font_data, Label_data, tag_prob, transform=None):
        self.transform = transform
        self.data_num = len(Font_data)
        self.tag_prob=tag_prob
        self.data = []
        self.label = []
        for i in range(self.data_num):
            x=torch.load(Font_data[i]).to("cpu").detach().numpy().copy()
            #mean.ver
            #y=(torch.mean(torch.load(Label_data[i]),0)).to("cpu").detach().numpy().copy()
            #random.ver
            y_=torch.load(Label_data[i])
            r=return_index(y_,tag_prob[i])
            y = y_[r[0]].to("cpu").detach().numpy().copy()
            self.data.append(x)
            self.label.append(y)
        self.data_num = len(self.data)

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        out_data = self.data[idx]
        out_label = self.label[idx]
        if self.transform:
            out_data = self.transform(out_data)

        return out_data, out_label


batch_size = 128
img_size=64
z_dim = 100
lambda_gp=10



class Generator(nn.Module):
    def __init__(self, z_dim, num_class):
        super(Generator, self).__init__()

        self.fc1 = nn.Linear(z_dim, 300)
        self.bn1 = nn.BatchNorm1d(300)
        self.LReLU1 = nn.LeakyReLU(0.2)

        self.fc2 = nn.Linear(num_class, 1500)
        self.bn2 = nn.BatchNorm1d(1500)
        self.LReLU2 = nn.LeakyReLU(0.2)

        self.fc3 = nn.Linear(1800, 128 * 16 * 16)
        self.bn3 = nn.BatchNorm1d(128 * 16 * 16)
        self.bo1 = nn.Dropout(p=0.5)
        self.LReLU3 = nn.LeakyReLU(0.2)

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), #チャネル数を128⇒64に変える。
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1), #チャネル数を64⇒1に変更
            nn.Tanh(),
        )

        self.init_weights()

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.ConvTranspose2d):
                module.weight.data.normal_(0, 0.02)
                module.bias.data.zero_()
            elif isinstance(module, nn.Linear):
                module.weight.data.normal_(0, 0.02)
                module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm1d):
                module.weight.data.normal_(1.0, 0.02)
                module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.normal_(1.0, 0.02)
                module.bias.data.zero_()

    def forward(self, noise, labels):
        y_1 = self.fc1(noise)
        y_1 = self.bn1(y_1)
        y_1 = self.LReLU1(y_1)

        y_2 = self.fc2(labels)
        y_2 = self.bn2(y_2)
        y_2 = self.LReLU2(y_2)
        x = torch.cat([y_1, y_2], 1)
        x = self.fc3(x)
        x = self.bo1(x)
        x = self.LReLU3(x)
        x = x.view(-1, 128, 16, 16)
        x = self.deconv(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, num_class):
        super(Discriminator, self).__init__()
        self.num_class = num_class

        self.conv = nn.Sequential(
            nn.Conv2d(num_class + 1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(128),
        )

        self.fc = nn.Sequential(
            nn.Linear(128 * 16 * 16, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024,1)
            #nn.Sigmoid()
        )

        self.init_weights()

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                module.weight.data.normal_(0, 0.02)
                module.bias.data.zero_()
            elif isinstance(module, nn.Linear):
                module.weight.data.normal_(0, 0.02)
                module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm1d):
                module.weight.data.normal_(1.0, 0.02)
                module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.normal_(1.0, 0.02)
                module.bias.data.zero_()

    def forward(self, img, labels):
        y_2 = labels.view(-1, self.num_class, 1, 1)
        y_2 = y_2.expand(-1, -1, img_size, img_size)
        x = torch.cat([img.float(), y_2], 1)

        x = self.conv(x)
        x = x.view(-1, 128 * 16 * 16)
        x = self.fc(x)
        return x

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

#Grand-penaltyの定義
def compute_gradient_penalty(D, real_samples, fake_samples, labels):
    """Calculates the gradient penalty loss for WGAN GP.
       Warning: It doesn't compute the gradient w.r.t the labels, only w.r.t
       the interpolated real and fake samples, as in the WGAN GP paper.
    """
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    labels = Tensor(labels)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates, labels)
    fake = Tensor(real_samples.shape[0], 1).fill_(1.0)
    fake.requires_grad = False
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )
    gradients = gradients[0].view(gradients[0].size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def train_func(D_model, G_model, batch_size, z_dim, num_class, criterion, 
               D_optimizer, G_optimizer, data_loader, device):
    #訓練モード
    D_model.train()
    G_model.train()

    #本物のラベルは1
    # y_real = torch.ones((batch_size, 1)).to(device)
    # D_y_real = (torch.rand((batch_size, 1))/2 + 0.7).to(device) #Dに入れるノイズラベル

    #偽物のラベルは0
    # y_fake = torch.zeros((batch_size, 1)).to(device)
    # D_y_fake = (torch.rand((batch_size, 1)) * 0.3).to(device) #Dに入れるノイズラベル

    #lossの初期化
    D_running_loss = 0
    G_running_loss = 0

    #バッチごとの計算
    for batch_idx, (data, labels) in enumerate(data_loader):
        #バッチサイズに満たない場合は無視
        batch_len=data.size()[0]
        # if data.size()[0] != batch_size:
        #     break

        #ノイズ作成
        z = torch.normal(mean = 0.5, std = 0.2, size = (batch_len, z_dim)) #平均0.5の正規分布に従った乱数を生成
        real_img, label, z = data.to(device), labels.to(device), z.to(device)
        label=label[:batch_len]
        #Discriminatorのc更新
        D_optimizer.zero_grad()
        if batch_idx%1==0:
            #Discriminatorに本物画像を入れて順伝播⇒Loss計算
            D_real = D_model(real_img, label)
            #D_real_loss = criterion(D_real, D_y_real)
            D_real_loss = torch.mean(D_real)
            #DiscriminatorにGeneratorにノイズを入れて作った画像を入れて順伝播⇒Loss計算
            fake_img = G_model(z, label)
            D_fake = D_model(fake_img.detach(), label)
            #D_fake_loss = criterion(D_fake, D_y_fake)
            D_fake_loss=torch.mean(D_fake)
            gradient_penalty = compute_gradient_penalty(D_model, real_img, fake_img ,label)
            #2つのLossの和を最小化
            #D_loss = D_real_loss + D_fake_loss
            #Wasserstein距離を計算し最小化
            D_loss=D_fake_loss-D_real_loss+lambda_gp * gradient_penalty
            D_loss.backward()
            D_optimizer.step()

            D_running_loss += D_loss.item()

        #Generatorの更新
        G_optimizer.zero_grad()
        if batch_idx%1==0:
            #Generatorにノイズを入れて作った画像をDiscriminatorに入れて順伝播⇒見破られた分がLossになる
            fake_img_2 = G_model(z, label)
            D_fake_2 = D_model(fake_img_2, label)

            #Gのloss(max(log D)で最適化)
            #G_loss = -criterion(D_fake_2, y_fake)
            G_loss= -torch.mean(D_fake_2)
            G_loss.backward()
            G_optimizer.step()
            G_running_loss += G_loss.item()

    D_running_loss /= len(data_loader)
    G_running_loss /= len(data_loader)

    return D_running_loss, G_running_loss

def Generate_img(epoch, G_model, device, z_dim, noise, var_mode, labels, log_dir = os.path.join("./result",dt_now,"logs_cWGAN")):
    G_model.eval()

    with torch.no_grad():
        if var_mode == True:
            #生成に必要な乱数
            noise = torch.normal(mean = 0.5, std = 0.2, size = (50, z_dim)).to(device)
        else:
            noise = noise

        #Generatorでサンプル生成
        samples = G_model(noise, labels).data.cpu()
        samples = (samples/2)+0.5
        save_image(samples,os.path.join(log_dir, 'epoch_%05d.png' % (epoch)), nrow = 10)


# In[11]:


#再現性確保のためseed値固定
SEED = 1111
random.seed(SEED)
np.random.seed(SEED) 
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
#device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def model_run(num_epochs, batch_size = batch_size, device = device):

    #Generatorに入れるノイズの次元
    var_mode = False #表示結果を見るときに毎回異なる乱数を使うかどうか
    #生成に必要な乱数
    noise = torch.normal(mean = 0.5, std = 0.2, size = (20, z_dim)).to(device)

    #クラス数
    num_class = 300

    #Generatorを試すときに使うラベルを作る
        #Generatorを試すときに使うラベルを作る
    target_label=test_Label[:20]
    target_image=test_Font[:20]
    target_labels=torch.cat([torch.mean(torch.load(l),0).view(-1,num_class) for l in target_label],0)
    target_images=torch.cat([torch.load(i) for i in target_image],0)
    label=target_labels
    target_label=None
    save_image(target_images.float().view(-1,1,img_size,img_size),os.path.join('./result/',dt_now,'logs_cWGAN/target.png'), nrow = 10)
    target_imges=None

    #モデル定義
    D_model = Discriminator(num_class).to(device)
    G_model = Generator(z_dim, num_class).to(device)
    #GPUの分散
    D_model=nn.DataParallel(D_model)
    G_model=nn.DataParallel(G_model)

    #lossの定義(引数はtrain_funcの中で指定)
    criterion = nn.BCELoss().to(device)

    #optimizerの定義
    D_optimizer = torch.optim.Adam(D_model.parameters(), lr=0.0002, betas=(0.5, 0.999), eps=1e-08, weight_decay=1e-5, amsgrad=False)
    G_optimizer = torch.optim.Adam(G_model.parameters(), lr=0.0002, betas=(0.5, 0.999), eps=1e-08, weight_decay=1e-5, amsgrad=False)

    D_loss_list = []
    G_loss_list = []

    all_time = time.time()
    for epoch in tqdm.tqdm(range(num_epochs),total=num_epochs):
        #         if (epoch + 1) == 6:
        #             D_optimizer.param_groups[0]['lr'] /= 10
        #             G_optimizer.param_groups[0]['lr'] /= 10
        #             print("learning rate change!")
        #         i(samples/2)+0.5f (epoch + 1) == 11:
        #             D_optimizer.param_groups[0]['lr'] /= 10
        #             G_optimizer.param_groups[0]['lr'] /= 10
        #             print("learning rate change!")
        start_time = time.time()
        train_data = dataset_full(train_Font, train_Label, tag_prob , transform=transform)
        dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
        D_loss, G_loss = train_func(D_model, G_model, batch_size, z_dim, num_class, criterion, 
                                    D_optimizer, G_optimizer, dataloader, device)

        D_loss_list.append(D_loss)
        G_loss_list.append(G_loss)

        secs = int(time.time() - start_time)
        mins = secs / 60
        secs = secs % 60

        #エポックごとに結果を表示
        print('Epoch: %d' %(epoch + 1), " | 所要時間 %d 分 %d 秒" %(mins, secs))
        print(f'\tLoss: {D_loss:.4f}(Discriminator)')
        print(f'\tLoss: {G_loss:.4f}(Generator)')

        if (epoch + 1) % 1 == 0:
            Generate_img(epoch, G_model, device, z_dim, noise, var_mode, label)

        #モデル保存のためのcheckpointファイルを作成
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch':epoch,
                'model_state_dict':G_model.state_dict(),
                'optimizer_state_dict':G_optimizer.state_dict(),
                'loss':G_loss,
            }, os.path.join('./result',dt_now,'checkpoint_cWGAN/G_model_{}'.format(epoch + 1)))

    return D_loss_list, G_loss_list
train_hist = {}
data_hist={}
data_hist["train"]=train_Font
data_hist["test"]=test_Font
train_hist['D_losses'], train_hist['G_losses'] = model_run(num_epochs = 5000)
pickle_dump(train_hist, os.path.join("./result/",dt_now,"train_hist.pickle"))
pickle_dump(data_hist,os.path.join("./result/",dt_now,"data_hist.pickle"))




