from car_network import RegistrationNet
from car_dataset import CARDataset, ToTensor
from torch.utils.data import DataLoader
from weight_regularization import Regularization
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from tqdm import tqdm
from datetime import datetime
import os
import math
import numpy as np
import json

seed = 6
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

root_dir = '/media/dell/_Files/simulation_data/all'
train_dir = root_dir + '/segment/train'
test_dir = root_dir + '/segment/test'
test_samples = os.listdir(test_dir + '/moving')

time = datetime.now().strftime('%Y_%m_%d_%H_%M')
save_dir = root_dir + '/training_files'
model_save_path = save_dir + '/models/' + time
summary_save_path = save_dir + '/summaries/' + time
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)
if not os.path.exists(summary_save_path):
    os.makedirs(summary_save_path)

point_num = 350
moving_channel = 3
target_channel = 2
train_batch_size = 1
test_batch_size = 1
save_iter = 3
num_epochs = 186
data_mode = 'spher'

train_data = CARDataset(dir=train_dir, point_num=point_num, data_mode=data_mode, transform=ToTensor)
train_dataloader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True, num_workers=4, pin_memory=True)
iter_num = math.ceil(len(train_data.sample_list) / train_batch_size) - 1

net = RegistrationNet(moving_channel=moving_channel, target_channel=target_channel).cuda()
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
    elif classname.find('Batch') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Instance') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
net.apply(weights_init)

optimizer = torch.optim.Adam(net.parameters(), lr=0.1)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[48 * len(train_dataloader), 135 * len(train_dataloader)], gamma=0.1)
writer = SummaryWriter(logdir=summary_save_path)

smooth_weight = 10
scale = 0.5
SDD = 1100
loss_reg = Regularization(net, weight_decay=1e-4, normalization_decay=1e-4, p=2).cuda()


def parameter_cal(diff):

    x = -diff[:, 0, :]
    y = -diff[:, 1, :]
    z = -diff[:, 2, :]

    length = torch.sqrt(torch.sum(torch.pow(diff, 2), dim=1))
    pro_length = torch.sqrt(torch.pow(x, 2) + torch.pow(y, 2))

    angle_one = torch.asin(pro_length / length)
    angle_one_mask_1 = torch.where(z > 0, torch.ones_like(angle_one), -torch.ones_like(angle_one))
    angle_one_mask_2 = torch.where(z > 0, torch.zeros_like(angle_one), math.pi * torch.ones_like(angle_one))
    angle_one = angle_one_mask_1 * angle_one + angle_one_mask_2

    angle_two = torch.asin(x / pro_length)
    angle_two_mask_1 = torch.where(y >= 0, -torch.ones_like(angle_two), torch.ones_like(angle_two))
    angle_two_mask_2 = torch.where(x > 0, -math.pi * torch.ones_like(angle_two), math.pi * torch.ones_like(angle_two))
    angle_two_mask_3 = torch.where(y >= 0, torch.zeros_like(angle_two), torch.ones_like(angle_two))
    angle_two = angle_two_mask_1 * angle_two + angle_two_mask_2 * angle_two_mask_3

    angle_ori = torch.stack((angle_one, angle_two), dim=1)

    return length, angle_ori / math.pi


def parameter_cal_2D(diff):

    x = -diff[:, 0, :]
    y = -diff[:, 1, :]

    length = torch.sqrt(torch.sum(torch.pow(diff, 2), dim=1))

    angle = torch.asin(x / length)
    angle_mask_1 = torch.where(y >= 0, -torch.ones_like(angle), torch.ones_like(angle))
    angle_mask_2 = torch.where(x > 0, -math.pi * torch.ones_like(angle), math.pi * torch.ones_like(angle))
    angle_mask_3 = torch.where(y >= 0, torch.zeros_like(angle), torch.ones_like(angle))
    angle = angle_mask_1 * angle + angle_mask_2 * angle_mask_3

    return length, angle / math.pi



if __name__ == '__main__':

    for epoch in range(1, num_epochs + 1):

        net.train()
        train_bar = tqdm(train_dataloader)

        half = torch.tril(torch.ones(point_num - 1, point_num - 1)).float().cuda()

        for iter, data in enumerate(train_bar):

            sample_norm = data['sample_norm'].squeeze(0).permute(0, 2, 1).float().cuda()
            sample = data['sample'].squeeze(0).permute(0, 2, 1).float().cuda()
            moving = sample[:, :3, :]
            target = sample[:, 3:, :]

            start_norm = data['start_norm'].squeeze(0).float().cuda()
            start = data['start'].squeeze(0).float().cuda()
            start = start.expand(-1, point_num - 1, -1)
            moving_start = start[:, :, :3]
            target_start = start[:, :, 3:]

            result = scale * net(sample_norm, start_norm)

            moving_length = moving[:, 0, :]
            moving_angle_ori = moving[:, 1:, :]
            moving_angle_new = moving_angle_ori + result
            moving_angle_new[:, 0, :] = 1 / (2 * scale + 1) * moving_angle_new[:, 0, :] + scale / (2 * scale + 1)
            moving_angle_new = math.pi * moving_angle_new

            moving_x = moving_length * torch.sin(moving_angle_new[:, 0, :]) * torch.sin(moving_angle_new[:, 1, :])
            moving_y = moving_length * torch.sin(moving_angle_new[:, 0, :]) * torch.cos(moving_angle_new[:, 1, :])
            moving_z = moving_length * torch.cos(moving_angle_new[:, 0, :])
            moving = torch.stack((moving_x, -moving_y, -moving_z), dim=-1)
            moving = moving_start + torch.matmul(half, moving)
            pro_x = moving[:, :, 0] / moving[:, :, 2] * SDD
            pro_y = moving[:, :, 1] / moving[:, :, 2] * SDD
            pro = torch.stack((pro_x, pro_y), dim=-1)

            target_length = target[:, 0, :]
            target_angle_ori = math.pi * target[:, 1, :]

            target_x = target_length * torch.sin(target_angle_ori)
            target_y = target_length * torch.cos(target_angle_ori)
            target = torch.stack((target_x, -target_y), dim=-1)
            target = target_start + torch.matmul(half, target)

            pro_loss = torch.sum(torch.sum(torch.pow(pro - target, 2), -1), -1)
            pro_loss /= point_num
            pro_loss = pro_loss.mean()

            temp = (result[:, :, :-1] - result[:, :, 1:])
            temp = temp[:, :, :-1] - temp[:, :, 1:]
            smooth_loss = torch.sum(torch.sum(torch.pow(temp, 2), -1), -1)
            smooth_loss = smooth_weight * smooth_loss.mean()

            weight_loss, normalization_loss = loss_reg(net)

            train_loss = pro_loss + smooth_loss + weight_loss + normalization_loss

            pro_loss = torch.sum(torch.sqrt(torch.sum(torch.pow(pro - target, 2), -1)), -1)
            pro_loss /= point_num
            pro_loss = pro_loss.mean()

            niter = (epoch - 1) * len(train_bar) + (iter + 1)
            writer.add_scalar('Training Loss', train_loss.item(), niter)
            writer.add_scalar('Registration Loss', pro_loss.item(), niter)
            writer.add_scalar('Smooth Loss', smooth_loss.item(), niter)
            writer.add_scalar('Weight Loss', weight_loss.item(), niter)
            writer.add_scalar('Normalization Loss', normalization_loss.item(), niter)

            train_bar.set_description(desc='[%d/%d] Pro Loss: %.4f  Smooth Loss: %.4f  W Loss: %.4f  In Loss: %.4f  lr: %.4f' %
                                     (epoch, num_epochs, pro_loss.item(), smooth_loss.item(),
                                      weight_loss.item(), normalization_loss.item(), optimizer.param_groups[0]['lr']))

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            scheduler.step()

        if epoch % save_iter == 0:

            state = {'model': net.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
            torch.save(state, model_save_path + '/' + 'epoch_' + str(epoch) + '.pth.tar')

            net.eval()

            with torch.no_grad():

                pro_sum = 0

                for sample in test_samples:

                    moving = torch.from_numpy(np.load(test_dir + '/moving/' + sample)).float().cuda()
                    target = torch.from_numpy(np.load(test_dir + '/target/' + sample)).float().cuda()
                    vessel_order = json.load(open(test_dir + '/parameter/' + sample[:-4] + '.json'))
                    segment_num = int(np.size(moving, 0) / point_num)

                    sample_mean = data['sample_mean'].unsqueeze(-1).expand(-1, -1, point_num - 1).float().cuda()
                    sample_std = data['sample_std'].unsqueeze(-1).expand(-1, -1, point_num - 1).float().cuda()
                    start_mean = data['start_mean'].unsqueeze(0).float().cuda()
                    start_std = data['start_std'].unsqueeze(0).float().cuda()

                    start_array = torch.zeros((1, segment_num, 3)).cuda()

                    seg_pro_loss = 0

                    for i in range(segment_num):

                        moving_seg = moving[point_num * i:point_num * (i + 1), :].permute(1, 0).unsqueeze(0)
                        if torch.sum(moving_seg) == 0:
                            continue
                        target_seg = target[point_num * i:point_num * (i + 1), :].permute(1, 0).unsqueeze(0)
                        seg = torch.cat((moving_seg, target_seg), dim=1)

                        moving_seg_dis = moving_seg[:, :, 1:] - moving_seg[:, :, :-1]
                        target_seg_dis = target_seg[:, :, 1:] - target_seg[:, :, :-1]
                        moving_length, moving_angle_ori = parameter_cal(moving_seg_dis)
                        target_length, target_angle_ori = parameter_cal_2D(target_seg_dis)
                        moving_spher = torch.cat((moving_length.unsqueeze(1), moving_angle_ori), dim=1)
                        target_spher = torch.cat((target_length.unsqueeze(1), target_angle_ori.unsqueeze(1)), dim=1)
                        seg_spher = torch.cat((moving_spher, target_spher), dim=1)

                        seg_spher_norm = (seg_spher - sample_mean) / sample_std

                        if vessel_order[str(i)] == None or vessel_order[str(i)] == 0:
                            moving_start = moving_seg[:, :, 0].unsqueeze(-1).permute(0, 2, 1)
                            start = seg[:, :, 0].unsqueeze(-1).permute(0, 2, 1)
                        else:
                            moving_start = start_array[:, vessel_order[str(i)], :].unsqueeze(1)
                            target_start = target_seg[:, :, 0].unsqueeze(-1).permute(0, 2, 1)
                            start = torch.cat((moving_start, target_start), dim=-1)
                        moving_start = moving_start.expand(-1, point_num - 1, -1)
                        start_norm = (start - start_mean) / start_std

                        result = scale * net(seg_spher_norm, start_norm)

                        moving_angle_new = moving_angle_ori + result
                        moving_angle_new[:, 0, :] = 1 / (2 * scale + 1) * moving_angle_new[:, 0, :] + scale / (2 * scale + 1)
                        moving_angle_new = math.pi * moving_angle_new

                        moving_x = moving_length * torch.sin(moving_angle_new[:, 0, :]) * torch.sin(moving_angle_new[:, 1, :])
                        moving_y = moving_length * torch.sin(moving_angle_new[:, 0, :]) * torch.cos(moving_angle_new[:, 1, :])
                        moving_z = moving_length * torch.cos(moving_angle_new[:, 0, :])
                        new_moving = torch.stack((moving_x, -moving_y, -moving_z), dim=-1)
                        new_moving = moving_start + torch.matmul(half, new_moving)

                        start_array[:, i, :] = new_moving[:, -1, :]

                        pro_x = new_moving[:, :, 0] / new_moving[:, :, 2] * SDD
                        pro_y = new_moving[:, :, 1] / new_moving[:, :, 2] * SDD
                        pro = torch.stack((pro_x, pro_y), dim=-1)

                        pro_loss = torch.sum(torch.sqrt(torch.sum(torch.pow(pro - target_seg[:, :, 1:].permute(0, 2, 1), 2), -1)), -1)
                        pro_loss /= point_num
                        pro_loss = pro_loss.cpu().numpy()
                        seg_pro_loss += pro_loss

                    seg_pro_average = seg_pro_loss / segment_num
                    pro_sum += seg_pro_average

                pro_average = pro_sum / len(test_samples)
                writer.add_scalar('Test Pro Loss', pro_average, epoch)
                print('pro loss: %2f' % (pro_average))

    writer.close()




