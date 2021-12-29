import torch
import os
import math
import numpy as np
import json
from car_network import RegistrationNet

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

point_num = 350
moving_channel = 3
target_channel = 2
scale = 0.5
SDD = 1100

test_dir = '/media/dell/_Files/simulation_data/all/segment/test'
save_dir = '/media/dell/_Files/simulation_data/all/segment/test/result'
parameter_dir = '/media/dell/_Files/aka/simulation_data/all/segment/train/parameter'
sample_list = os.listdir(test_dir + '/moving')

moving_mean = torch.from_numpy(np.load(parameter_dir + '/moving_spher_mean.npy'))
moving_std = torch.from_numpy(np.load(parameter_dir + '/moving_spher_std.npy'))
target_mean = torch.from_numpy(np.load(parameter_dir + '/target_spher_mean.npy'))
target_std = torch.from_numpy(np.load(parameter_dir + '/target_spher_std.npy'))
moving_start_mean = torch.from_numpy(np.load(parameter_dir + '/moving_start_mean.npy'))
moving_start_std = torch.from_numpy(np.load(parameter_dir + '/moving_start_std.npy'))
target_start_mean = torch.from_numpy(np.load(parameter_dir + '/target_start_mean.npy'))
target_start_std = torch.from_numpy(np.load(parameter_dir + '/target_start_std.npy'))

sample_mean = torch.cat((moving_mean, target_mean)).unsqueeze(0).unsqueeze(-1).expand(-1, -1, point_num - 1).float().cuda()
sample_std = torch.cat((moving_std, target_std)).unsqueeze(0).unsqueeze(-1).expand(-1, -1, point_num - 1).float().cuda()
start_mean = torch.cat((moving_start_mean, target_start_mean)).unsqueeze(0).float().cuda()
start_std = torch.cat((moving_start_std, target_start_std)).unsqueeze(0).float().cuda()

half = torch.tril(torch.ones(point_num - 1, point_num - 1)).float().cuda()

net = RegistrationNet(moving_channel=moving_channel, target_channel=target_channel).cuda()
checkpoint = torch.load('/media/dell/_Files/simulation_data/all/training_files/models/2021_10_07_17_41/epoch_186.pth.tar')
net.load_state_dict(checkpoint['model'])
net.eval()

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

    with torch.no_grad():

        pro_sum = 0

        for sample in sample_list:

            moving = torch.from_numpy(np.load(test_dir + '/moving/' + sample)).float().cuda()
            target = torch.from_numpy(np.load(test_dir + '/target/' + sample)).float().cuda()
            vessel_order = json.load(open(test_dir + '/parameter/' + sample[:-4] + '.json'))
            segment_num = int(np.size(moving, 0) / point_num)

            start_array = torch.zeros((1, segment_num, 3)).cuda()
            pro_array = torch.zeros(point_num * segment_num, 2)
            deformed_moving_array = torch.zeros(point_num * segment_num, 3)

            seg_pro_loss = 0

            for i in range(segment_num):

                moving_seg = moving[point_num * i:point_num * (i + 1), :].permute(1, 0).unsqueeze(0)
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
                a = moving_start
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
                new_moving = torch.cat((a, new_moving), dim=1)
                deformed_moving_array[point_num * i:point_num * (i + 1), :] = new_moving

                start_array[:, i, :] = new_moving[:, -1, :]

                pro_x = new_moving[:, :, 0] / new_moving[:, :, 2] * SDD
                pro_y = new_moving[:, :, 1] / new_moving[:, :, 2] * SDD
                pro = torch.stack((pro_x, pro_y), dim=-1)
                pro_array[point_num * i:point_num * (i + 1), :] = pro

                pro_loss = torch.sum(torch.sqrt(torch.sum(torch.pow(pro - target_seg.permute(0, 2, 1), 2), -1)), -1)
                pro_loss /= point_num
                pro_loss = pro_loss.cpu().numpy()
                seg_pro_loss += pro_loss

            seg_pro_average = seg_pro_loss / segment_num
            np.save(save_dir + '/loss/' + sample, seg_pro_average)
            pro_sum += seg_pro_average

            pro_array = pro_array.cpu().numpy()
            np.save(save_dir + '/moving_pro/' + sample, pro_array)

            deformed_moving_array = deformed_moving_array.cpu().numpy()
            np.save(save_dir + '/moving/' + sample, deformed_moving_array)

        pro_average = pro_sum / len(sample_list)
        print('pro loss: %2f' % (pro_average))

















