import os.path
from data.base_dataset import BaseDataset, get_params, get_transform , CoarseMatch , get_coarse_params
from data.image_folder import make_dataset
from PIL import Image
import random
import torchvision.transforms as transforms
import torch


class KittiflowDataset(BaseDataset):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = BaseDataset.modify_commandline_options(parser, is_train)
        parser.add_argument('--patch_size', type=int, default=7, help='kernel size for feature correlation')
        parser.add_argument('--search_size', type=int, default=11 , help='search steps for feature correlation')
        parser.add_argument('--search_stride', type=int, default=2, help='stride for feature correlation')
        parser.add_argument('--coarse_stride', type=int, default=5, help='stride for coarse match')
        parser.add_argument('--coarse_patch_size', type=int, default=50, help='kernel size for coarse match')
        parser.add_argument('--coarse_maxsize_w', type=int, default=150, help='stride for coarse match')
        parser.add_argument('--coarse_maxsize_h', type=int, default=30, help='stride for coarse match')
        parser.add_argument('--coarse_num_w', type=int, default=10, help='stride for coarse match')
        parser.add_argument('--coarse_num_h', type=int, default=6, help='stride for coarse match')
        parser.set_defaults(preprocess_mode='crop')
        parser.set_defaults(load_size=286)
        parser.set_defaults(crop_size=256) # crop_size must be the scale of 32
        parser.set_defaults(patch_size=7)
        parser.set_defaults(search_size=11)
        parser.set_defaults(search_stride=2)
        parser.set_defaults(display_winsize=256)
        parser.set_defaults(label_nc=3)

        return parser

    def initialize(self, opt):
        self.opt = opt

        self.dir_AB = os.path.join(opt.dataroot, 'origin_1')
        self.dir_sem = os.path.join(opt.dataroot, 'mask_semantic')
        self.AB_paths = sorted(make_dataset(self.dir_AB))
        self.sem_paths = sorted(make_dataset(self.dir_sem))

        size = len(self.AB_paths)
        self.dataset_size = size-1
        self.content_crop_size = opt.crop_size
        self.cps = opt.patch_size + opt.search_stride * (opt.search_size - 1)
        self.search_pad = (self.cps - opt.patch_size) // 2
        self.crop_size = opt.crop_size
        self.load_size  = self.crop_size + 2 * self.search_pad


        self.dir_CD = os.path.join(opt.dataroot,'origin_2')
        self.CD_paths = sorted(make_dataset(self.dir_CD))

        self.dir_AB_ref = os.path.join(opt.dataroot,'origin_1_ref')
        self.dir_CD_ref = os.path.join(opt.dataroot,'origin_2_ref')

        self.dir_real_sem_1 = os.path.join(opt.dataroot,'real_semantic_view_1')

        self.dir_real_sem_2 = os.path.join(opt.dataroot, 'real_semantic_view_2')



    def __getitem__(self, index):


        if index == self.dataset_size - 1 :
            index = index - 1


        AB_path = self.AB_paths[index % self.dataset_size + 1]
        image_path = AB_path
        AB = Image.open(AB_path).convert('RGB')
        w, h = AB.size
        h2 = int(h / 2)

        real_img = AB.crop((0, h2, w, h))

        image_str = self.AB_paths[index % self.dataset_size + 1]
        image_int = int(image_str[len(image_str)-10:len(image_str)-4])

        image_int_bef = image_int - 1
        image_int_aft = image_int + 1

        image_str_bef = str(image_int_bef).zfill(6) + '.png'
        image_str_aft = str(image_int_aft).zfill(6) + '.png'
        image_str_now = str(image_int).zfill(6) + '.png'

        condition_path = self.dir_AB_ref + '/' + image_str_bef
        condition = Image.open(condition_path).convert('RGB')
        condition = condition.crop((0, h2, w, h))

        condition_path_2 = self.dir_AB_ref + '/' + image_str_aft
        condition_2 = Image.open(condition_path_2).convert('RGB')
        condition_2 = condition_2.crop((0, h2, w, h))

        condition_view_path = self.dir_CD_ref + '/' + image_str_bef
        condition_view = Image.open(condition_view_path).convert('RGB')

        condition_view_path_2 = self.dir_CD_ref + '/' + image_str_aft
        condition_view_2 = Image.open(condition_view_path_2).convert('RGB')

        if not self.opt.test_mode:
            mask_index = random.randint(0, self.dataset_size)
            mask_path = self.sem_paths[mask_index % self.dataset_size]
        else:
            mask_path = self.sem_paths[index % self.dataset_size + 1]

        mask = Image.open(mask_path).convert('RGB')
        sem_tensor = transforms.ToTensor()(mask)
        img_tensor = transforms.ToTensor()(real_img)
        mask_tensor = torch.where(sem_tensor>0, img_tensor, sem_tensor)
        content = transforms.ToPILImage()(mask_tensor)

        params = get_params(self.opt, self.load_size, condition.size)

        if self.load_size > condition.size[1]:
            condition_size = [self.crop_size + 2 * self.search_pad, 320 + 2 * self.search_pad]
            content_size = [self.crop_size, 320]
        else:
            condition_size = [self.crop_size + 2 * self.search_pad, self.crop_size + 2 * self.search_pad]
            content_size = [self.crop_size, self.crop_size]

        content_transforms = get_transform(self.opt, params, content_size)

        if self.opt.use_coarse_match:

            condition_tensor = transforms.ToTensor()(condition)
            condition_2_tensor = transforms.ToTensor()(condition_2)
            condition_view_tensor = transforms.ToTensor()(condition_view)
            condition_view_2_tensor = transforms.ToTensor()(condition_view_2)

            coarse_match_condition = CoarseMatch(self.opt, mask_tensor, sem_tensor, condition_tensor)
            coarse_match_condition_2 = CoarseMatch(self.opt, mask_tensor, sem_tensor, condition_2_tensor)
            coarse_match_condition_view = CoarseMatch(self.opt, mask_tensor, sem_tensor, condition_view_tensor)
            coarse_match_condition_view_2 = CoarseMatch(self.opt, mask_tensor, sem_tensor, condition_view_2_tensor)

            condition_params = get_coarse_params(params, coarse_match_condition, content_size)
            condition_2_params = get_coarse_params(params, coarse_match_condition_2, content_size)
            condition_view_params = get_coarse_params(params, coarse_match_condition_view, content_size)
            condition_view_2_params = get_coarse_params(params, coarse_match_condition_view_2, content_size)

            condition_transforms = get_transform(self.opt, condition_params, condition_size)
            condition_2_transforms = get_transform(self.opt, condition_2_params, condition_size)
            condition_view_transforms = get_transform(self.opt, condition_view_params, condition_size)
            condition_view_2_transforms = get_transform(self.opt, condition_view_2_params, condition_size)

        if not self.opt.use_coarse_match:
            condition_transforms = get_transform(self.opt, params, condition_size)
            condition_2_transforms = get_transform(self.opt, params, condition_size)
            condition_view_transforms = get_transform(self.opt, params, condition_size)
            condition_view_2_transforms = get_transform(self.opt, params, condition_size)

        mask_transforms = get_transform(self.opt, params, content_size, normalize=False)
        real_image = content_transforms(real_img)
        content = content_transforms(content)
        mask = mask_transforms(mask) * 255.0

        condition = condition_transforms(condition)
        condition_2 = condition_2_transforms(condition_2)
        condition_view = condition_view_transforms(condition_view)
        condition_view_2 = condition_view_2_transforms(condition_view_2)

        condition_semantic_list = torch.zeros_like(condition)[0, :,:].unsqueeze(0)
        content_semantic_tensor = torch.zeros_like(content)

        select_list_str = list(self.opt.select_list)
        select_list_int = [int(num) for num in select_list_str]

        if self.opt.use_semantic:

            content_semantic_path = self.dir_real_sem_1 + '/' + image_str_now

            condition_semantic_path = self.dir_real_sem_1 + '/' + image_str_bef
            condition_semantic_path_2 = self.dir_real_sem_1 + '/' + image_str_aft

            condition_view_semantic_path = self.dir_real_sem_2 + '/' + image_str_bef
            condition_view_semantic_path_2 = self.dir_real_sem_2 + '/' + image_str_aft

            content_semantic = Image.open(content_semantic_path).convert('RGB')

            condition_semantic_1 = Image.open(condition_semantic_path).convert('RGB')
            condition_semantic_2 = Image.open(condition_semantic_path_2).convert('RGB')

            condition_view_semantic_1 = Image.open(condition_view_semantic_path).convert('RGB')
            condition_view_semantic_2 = Image.open(condition_view_semantic_path_2).convert('RGB')

            content_semantic_tensor = content_transforms(content_semantic)[ 0, :,:].unsqueeze(0)
            condition_semantic_tensor = condition_transforms(condition_semantic_1)[ 0, :,:].unsqueeze(0)
            condition_semantic_2_tensor = condition_transforms(condition_semantic_2)[ 0, :,:].unsqueeze(0)
            condition_view_semantic_1_tensor = condition_transforms(condition_view_semantic_1)[ 0, :,:].unsqueeze(0)
            condition_view_semantic_2_tensor = condition_transforms(condition_view_semantic_2)[ 0, :,:].unsqueeze(0)

            condition_semantic_all_list = [condition_semantic_tensor, condition_semantic_2_tensor, condition_view_semantic_1_tensor, condition_view_semantic_2_tensor]


            condition_semantic_list = condition_semantic_all_list[0]

            if len(select_list_int) > 1:

                for idx in range(len(select_list_int)):
                    if idx == 0:
                        continue
                    else:
                        condition_semantic_list = torch.cat([condition_semantic_list,condition_semantic_all_list[select_list_int[idx]]], dim=0)

        condition_all_list = [condition, condition_2, condition_view, condition_view_2]

        condition_list = condition_all_list[0]

        if len(select_list_int) > 1:

            for idx2 in range(len(select_list_int)):
                if idx2 == 0:
                    continue
                else:
                    condition_list = torch.cat([condition_list, condition_all_list[select_list_int[idx2]]], dim=0)

        input_dict = {'label': content,
                      'mask': mask,
                      'condition_list': condition_list,
                      'content_semantic': content_semantic_tensor,
                      'condition_semantic_list': condition_semantic_list,
                      'image': real_image,
                      'path': image_path,
                      }

        self.postprocess(input_dict)

        return input_dict

    def postprocess(self, input_dict):
        return input_dict

    def __len__(self):
        return self.dataset_size


