import random
import json
from decord import VideoReader, cpu
import cv2
import torch
import os
import numpy as np
from PIL import Image
import einops
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
import csv
import math
import copy
import requests
from io import BytesIO
# from transformers import CLIPImageProcessor
from torchvision.transforms.functional import resize, center_crop, pil_to_tensor
from diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion \
    import _resize_with_antialiasing, _append_dims
# from transformers import SiglipImageProcessor
# from dataloading.constants import ASPECT_RATIO
# ASPECT_RATIO = 1/1


ASPECT_RATIO_960 = {
    '0.25': [480., 1920.], '0.26': [480., 1856.], '0.27': [480., 1792.], '0.28': [480., 1728.],
    '0.32': [544., 1728.], '0.33': [544., 1664.], '0.35': [544., 1600.], '0.4':  [608., 1536.],
    '0.42':  [608., 1472.], '0.48': [672., 1408.], '0.5': [672., 1344.], '0.52': [672., 1280.],
    '0.57': [736., 1280.], '0.6': [736., 1216.], '0.68': [800., 1152.], '0.72': [800., 1088.],
    '0.78': [864., 1088.], '0.82': [864.,  960.], '0.88': [928.,  960.], '0.94': [928.,  896.],
    '1.0':  [960.,  960.], '1.07': [960.,  896.], '1.13': [1024.,  896.], '1.21': [1024.,  832.],
    '1.29': [1088.,  832.], '1.38': [1088.,  768.], '1.46': [1152.,  768.], '1.67': [1216.,  704.],
    '1.75': [1280.,  704.], '2.0': [1344.,  640.], '2.09': [1408.,  640.], '2.4': [1472.,  576.],
    '2.5': [1536.,  576.], '2.89': [1600.,  544.], '3.0': [1664.,  544.], '3.11': [1728.,  544.],
    '3.62': [1792.,  512.], '3.75': [1856.,  512.], '3.88': [1920.,  512.], '4.0': [1920.,  480.],
}


def get_closest_ratio(height: float, width: float, ratios: dict):
    aspect_ratio = float(height / width)
    closest_ratio = min(ratios.keys(), key=lambda ratio: abs(float(ratio) - aspect_ratio))
    return ratios[closest_ratio], float(closest_ratio)

@torch.no_grad()
def find_flat_region(mask, dtype):
    device = mask.device
    kernel_x = torch.Tensor([[-1, 0, 1], [-1, 0, 1],
                             [-1, 0, 1]]).unsqueeze(0).unsqueeze(0).to(device, dtype)
    kernel_y = torch.Tensor([[-1, -1, -1], [0, 0, 0],
                             [1, 1, 1]]).unsqueeze(0).unsqueeze(0).to(device, dtype)
    mask_ = F.pad(mask.unsqueeze(0), (1, 1, 1, 1), mode='replicate')

    grad_x = torch.nn.functional.conv2d(mask_, kernel_x)
    grad_y = torch.nn.functional.conv2d(mask_, kernel_y)
    return ((abs(grad_x) + abs(grad_y)) == 0).float()[0].to(dtype=dtype)

class BaseDataset(Dataset):
    def __init__(self, args):
        self.args = args
        self.repeat = 1

        # self.img_size = args.resolution
        self.nframes = args.nframes # add ref_img
        self.csv_file_list = args.csv_file_list
        self.mask_csv_file_list = args.mask_csv_file_list
        self.data_root = args.data_root
        self._load_metadata()
        self.mask_len = len(self.metadata_mask)
        print('len of metadata: ', len(self.metadata))

        # self.transform = transforms.Compose([
        #     transforms.Resize(self.img_size, interpolation=transforms.InterpolationMode.BICUBIC),
        #     transforms.ToTensor(),
        #     transforms.Normalize([0.5], [0.5]),
        # ])  # [-1, 1]

        # self.feature_extractor = SiglipImageProcessor.from_pretrained(args.siglip_path)
        # self.clip_image_processor = transforms.Lambda(self._encode_image)
        
        # self.cond_transform = transforms.Compose([
        #     transforms.Resize(self.img_size, interpolation=transforms.InterpolationMode.BILINEAR),
        #     transforms.ToTensor(),
        #     transforms.Normalize([0.5], [0.5]),
        # ])  # [-1, 1]  w.r.t controlnet cond_image

        # self.mask_transform = transforms.Compose([
        #     transforms.Resize(self.img_size, interpolation=transforms.InterpolationMode.BILINEAR),
        #     transforms.ToTensor(),
        #     # transforms.Normalize([0.5], [0.5]),
        # ])

    def _encode_image(self, image_pixels):
        image = pil_to_tensor(image_pixels) / 255.0  # [0, 1]

        image = image * 2.0 - 1.0  # [-1, 1]
        image = _resize_with_antialiasing(image[None], (224, 224))
        image = (image + 1.0) / 2.0  # [0, 1]

        # Normalize the image with for CLIP input
        image = self.feature_extractor(
                images=image,
                do_normalize=True,
                do_center_crop=False,
                do_resize=False,
                do_rescale=False,
                return_tensors="pt",
            ).pixel_values
        return image

    def preprocess(self, image_pixels, ASPECT_RATIO):
        w, h = image_pixels.size
        if h>w:
            w_target, h_target = self.img_size, int(self.img_size / ASPECT_RATIO // 64) * 64
        else:
            w_target, h_target = int(self.img_size / ASPECT_RATIO // 64) * 64, self.img_size
        h_w_ratio = float(h) / float(w)
        if h_w_ratio < h_target / w_target:
            h_resize, w_resize = h_target, math.ceil(h_target / h_w_ratio)
        else:
            h_resize, w_resize = math.ceil(w_target * h_w_ratio), w_target
        image_pixels = resize(image_pixels, [h_resize, w_resize], antialias=None)
        image_pixels = center_crop(image_pixels, [h_target, w_target])
        return image_pixels


    def _aug(self, frame, transform1, transform2=None, state=None):
        if state is not None:
            torch.set_rng_state(state)
        frame_transform1 = transform1(frame) if transform1 is not None else frame
        if transform2 is None:
            return frame_transform1
        else:
            return transform2(frame_transform1)
    

    def _load_metadata(self):
        self.metadata = []
        for csv_file in self.csv_file_list:
            with open(csv_file, 'r', encoding="utf-8") as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    self.metadata.append(row)
        self.metadata_mask = []
        for csv_file in self.mask_csv_file_list:
            with open(csv_file, 'r', encoding="utf-8") as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    self.metadata_mask.append(row)
    
    def __len__(self):
        return len(self.metadata) * self.repeat
    
    def _mask_hw(self, height_v, width_v, height_m, width_m):
        if height_v>height_m:
            h_l = random.randint(0, height_v-height_m-1)
            h_temp = height_m + h_l
            pad_h = True
        elif height_v<height_m:
            h_l = random.randint(0, height_m-height_v-1)
            h_temp = height_v + h_l
            pad_h = False
        else:
            h_l = 0
            h_temp = height_v
            pad_h = False

        if width_v>width_m:
            w_l = random.randint(0, width_v-width_m-1)
            w_temp = width_m + w_l
            pad_w = True
        elif width_v<width_m:
            w_l = random.randint(0, width_m-width_v-1)
            w_temp = width_v + w_l
            pad_w = False
        else:
            w_l = 0
            w_temp = width_v
            pad_w = False

        return [pad_h, pad_w], [h_l, h_temp, w_l, w_temp]
    
    def _mask_hw_resize(self, height_v, width_v, ori_ratio, mask_ratio):
        if ori_ratio > mask_ratio:
            mask_resize_size = int(width_v * mask_ratio), width_v
            pad = "h"
            l_start = random.randint(0, height_v-mask_resize_size[0]-1)
            pad_temp = mask_resize_size[0] + l_start
        elif ori_ratio < mask_ratio:
            mask_resize_size = height_v, int(height_v / mask_ratio)
            l_start = random.randint(0, width_v-mask_resize_size[1]-1)
            pad_temp = mask_resize_size[1] + l_start
            pad = "w"
        else:
            mask_resize_size = height_v, width_v
            pad = "z"
            l_start = 0
            pad_temp = width_v
        mask_resize_size = [mask_resize_size[1], mask_resize_size[0]] # for cv2 resize
        return [mask_resize_size, pad], [l_start, pad_temp]
    
    def _mask_padding(self, mask, pad_flag, hw, height_v, width_v, height_m, width_m):
        if pad_flag[0]:
            mask_temp_h = np.zeros((height_v, width_m, 3))
            mask_temp_h[hw[0]:hw[1], :, :] = mask
        else:
            mask_temp_h = mask[hw[0]:hw[1], :, :]
        if pad_flag[1]:
            mask_temp = np.zeros((height_v, width_v, 3))
            mask_temp[:, hw[2]:hw[3], :] = mask_temp_h
        else:
            mask_temp = mask_temp_h[:, hw[2]:hw[3], :]

        return mask_temp
        
    def _mask_padding_resize(self, mask, mask_res_d, hw, height_v, width_v):
        mask = cv2.resize(mask, mask_res_d[0])
        if mask_res_d[1] == "h":
            mask_temp = np.zeros((height_v, width_v, 3))
            mask_temp[hw[0]:hw[1], :, :] = mask
        elif mask_res_d[1] == "w":
            mask_temp = np.zeros((height_v, width_v, 3))
            mask_temp[:, hw[0]:hw[1], :] = mask
        else:
            mask_temp = mask
        return mask_temp
    
    def __getitem__(self, index):
        # ["width", "height", "duration", "vid_path", "caption", "url"]
        while True:
            index = index % len(self.metadata)

            # read video
            raw_data = self.metadata[index]
            index_mask = random.randint(0, self.mask_len - 1)
            raw_mask_data = self.metadata_mask[index_mask]
            #vid_path = os.path.join(self.data_root, raw_data['vid_path'])
            vid_path = os.path.join(self.data_root, raw_data['video_path'])
            vid_caption = raw_data['prompt']
            # vid_caption = raw_data['caption']
            # text_embeds = np.load(BytesIO(requests.get(raw_data['text_embeds']).content))
            sam2_path = os.path.join(self.data_root, raw_mask_data["mask_path"])

            try:
                video_reader = VideoReader(vid_path)
                sam2_reader = VideoReader(sam2_path)
                if len(video_reader) < self.nframes:
                    repeat_num_v = math.ceil(self.nframes / len(video_reader))
                    # print(f"video length ({len(video_reader)}) is smaller than target length({self.n_sample_frames})")
                    # index += 1
                    # continue
                else:
                    repeat_num_v = 0
                if len(sam2_reader) < self.nframes:
                    repeat_num = math.ceil(self.nframes / len(sam2_reader))
                    # print(f"mask video length ({len(sam2_reader)}) is smaller than target length({self.n_sample_frames})")
                    # index += 1
                    # continue
                else:
                    repeat_num = 0
                
                # frame_stride = random.randint(1, 6)
                frame_stride = 1
                fs = 1
                
                if frame_stride != 1:
                    all_frames = list(range(0, len(video_reader), frame_stride))
                    mask_all_frames = list(range(0, len(sam2_reader), frame_stride))
                    if len(all_frames) < self.nframes:
                        fs = len(video_reader) // self.nframes
                        assert(fs != 0)
                        all_frames = list(range(0, len(video_reader), fs))
                else:
                    if repeat_num_v == 0:
                        all_frames = list(range(len(video_reader)))
                    else:
                        #temp_list = list(range(len(video_reader))) + list(range(len(video_reader)))[::-1][1:-1]
                        #all_frames = temp_list * repeat_num_v
                        if random.random()>=0.5:
                            temp_list = list(range(len(video_reader))) + list(range(len(video_reader)))[::-1][1:-1]
                            all_frames = temp_list * repeat_num_v
                        else:
                            all_frames = list(range(len(video_reader))) + [list(range(len(video_reader)))[-1]] * (self.nframes - len(video_reader) + 3)
                    if repeat_num==0:
                        mask_all_frames = list(range(len(sam2_reader)))
                    else:
                        #temp_list = list(range(len(sam2_reader))) + list(range(len(sam2_reader)))[::-1][1:-1]
                        #mask_all_frames = temp_list * repeat_num
                        if random.random()>=0.5:
                            temp_list = list(range(len(sam2_reader))) + list(range(len(sam2_reader)))[::-1][1:-1]
                            mask_all_frames = temp_list * repeat_num
                        else:
                            mask_all_frames = list(range(len(sam2_reader))) + [list(range(len(sam2_reader)))[-1]] * (self.nframes - len(sam2_reader) + 3)

                # select a random clip
                rand_idx = random.randint(0, max(0, len(all_frames) - self.nframes-1))
                frame_indices = all_frames[rand_idx: rand_idx + self.nframes]
                mask_rand_idx = random.randint(0, max(0, len(mask_all_frames) - self.nframes-1))
                mask_frame_indices = mask_all_frames[mask_rand_idx: mask_rand_idx + self.nframes]
            except:
                index += 1
                print(f"Load video failed! path={vid_path}")
                continue

            '''stage 2
            try:
                if len(frame_indices)<self.nframes or len(mask_frame_indices)<self.nframes:
                    print(f"vid frames are {len(frame_indices)}, mask frames are {len(mask_frame_indices)}")
                    index += 1
                    continue
                if random.random()>=0.8:
                    mask_frame_indices_new = [mask_frame_indices[random.randint(0, len(mask_frame_indices)-1)]] * len(mask_frame_indices)
                    if sam2_path.endswith(".mp4"):
                        masks = sam2_reader.get_batch(mask_frame_indices_new).asnumpy()
                    else:
                        mask = decode(sam2_reader[mask_frame_indices_new[0]][mask_select])
                        mask = np.repeat(np.expand_dims(mask, axis=2), repeats=3, axis=2)
                        masks = np.repeat(np.expand_dims(mask, axis=0), repeats=len(mask_frame_indices_new), axis=0) * 255
                    if random.random()>=0.7:
                        frame_indices_new = [frame_indices[random.randint(0, len(frame_indices)-1)]] * len(frame_indices)
                        video = video_reader.get_batch(frame_indices_new).asnumpy()
                    else:
                        video = video_reader.get_batch(frame_indices).asnumpy()  # [f h w c]
                else:
                    if sam2_path.endswith(".mp4"):
                        masks = sam2_reader.get_batch(mask_frame_indices).asnumpy()
                    else:
                        masks = np.array([decode(sam2_reader[mask_frame_indices[frm_idx]][mask_select]) * 255 for frm_idx in range(len(mask_frame_indices))])
                        masks = np.repeat(np.expand_dims(masks, axis=-1), repeats=3, axis=-1)
                    if random.random()>=0.9:
                        frame_indices_new = [frame_indices[random.randint(0, len(frame_indices)-1)]] * len(frame_indices)
                        video = video_reader.get_batch(frame_indices_new).asnumpy()
                    else:
                        video = video_reader.get_batch(frame_indices).asnumpy()  # [f h w c]
            
                fps = video_reader.get_avg_fps() * fs
                height_v, width_v, _ = video[0].shape
                if height_v<width_v:
                    if random.random()>=0.6:
                        width_v_new = int(height_v //4 * 3)
                        width_v_start = random.randint(0, (width_v - width_v_new - 1))
                        video = video[:, :, width_v_start:(width_v_start+width_v_new), :]
                break
            except:
                print(f"Get frames failed! path = {vid_path}")
                index += 1
                continue
            '''
            # stage 1
            try:
                if len(frame_indices)<self.nframes or len(mask_frame_indices)<self.nframes:
                    print(f"vid frames are {len(frame_indices)}, mask frames are {len(mask_frame_indices)}")
                    index += 1
                    continue
                video = video_reader.get_batch(frame_indices).asnumpy()  # [f h w c]
                masks = sam2_reader.get_batch(mask_frame_indices).asnumpy()
                fps = video_reader.get_avg_fps() * fs
                break
            except:
                print(f"Get frames failed! path = {vid_path}")
                index += 1
                continue

        assert (video.shape[0] == self.nframes), f'{video.shape[0]}, self.nframes={self.nframes}'

        input_imgs        = []
        input_masked_imgs = []
        input_masks       = []
        height_v, width_v, _ = video[0].shape
        height_m, width_m, _ = masks[0].shape
        ori_ratio  = height_v/width_v
        mask_ratio = height_m/width_m
        # pad_flag, hw = self._mask_hw(height_v, width_v, height_m, width_m)
        mask_res_d, hw = self._mask_hw_resize(height_v, width_v, ori_ratio, mask_ratio)

        # first check the size of the input frame

        closest_size, closest_ratio = get_closest_ratio(height_v, width_v, ASPECT_RATIO_960)  # TODO
        closest_size = list(map(lambda x: int(x), closest_size))
        
        # if closest_size[0] / height_v > closest_size[1] / width_v:
        #     resize_size = closest_size[0], int(width_v * closest_size[0] / height_v)
        # else:
        #     resize_size = int(height_v * closest_size[1] / width_v), closest_size[1]
        if closest_ratio > ori_ratio:
            resize_size = height_v, int(height_v / closest_ratio)
        else:
            resize_size = int(width_v * closest_ratio), width_v

        
        self.transform = transforms.Compose([
            # transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BICUBIC),
            # transforms.CenterCrop(closest_size),
            transforms.RandomCrop(resize_size),
            transforms.Resize(closest_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])  # [-1, 1]

        self.mask_transform = transforms.Compose([
            # transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.NEAREST),
            # transforms.CenterCrop(closest_size),
            transforms.RandomCrop(resize_size),
            transforms.Resize(closest_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
        ])  # [0, 1]

        # transform1 = transforms.Lambda(self.preprocess)
        state = torch.get_rng_state()
        for t in range(self.nframes):
            frame_temp = video[t]

            img = Image.fromarray(frame_temp)
            # input_img = self._aug(img, transform1, self.transform, state)  # crop to square, [-1, 1], for vae
            input_img = self._aug(img, self.transform, None, state)  # crop to square, [-1, 1], for vae
            input_imgs.append(input_img)

            # bg
            frame_temp_bg = copy.deepcopy(frame_temp)
            mask_temp = masks[t]
            # mask_temp = self._mask_padding(mask_temp, pad_flag, hw, height_v, width_v, height_m, width_m)
            mask_temp = self._mask_padding_resize(mask_temp, mask_res_d, hw, height_v, width_v)
            mask_temp = (mask_temp > 5).astype(np.uint8)
            frame_temp_bg = frame_temp_bg * (1- mask_temp)
            frame_temp_bg = Image.fromarray(frame_temp_bg)
            # frame_temp_bg = self._aug(frame_temp_bg, transform1, self.transform, state)  # crop to square, [-1, 1], for vae
            frame_temp_bg = self._aug(frame_temp_bg, self.transform, None, state)
            # aa = find_flat_region(frame_temp_bg[0, ...], frame_temp_bg.dtype)
            input_masked_imgs.append(frame_temp_bg)

            # mask_temp = Image.fromarray(mask_temp*255)
            # mask_temp = mask_temp.convert("L")
            # mask_temp = self._aug(mask_temp, transform1, self.mask_transform, state)  # [0, 1], for bg seg
            # input_masks.append(mask_temp)
            
            '''if t == 0:
                mask_max = np.zeros_like(mask_temp)
                mask_temp = Image.fromarray(mask_temp*255)
                mask_temp = mask_temp.convert("L")
                # mask_temp = self._aug(mask_temp, transform1, self.mask_transform, state)  # [0, 1], for bg seg
                mask_temp = self._aug(mask_temp, self.mask_transform, None, state)
                input_masks.append(mask_temp)
            # wan vae downsample 4x
            elif t%4 == 0:
                mask_max = mask_max + mask_temp
                mask_max = np.array(mask_max>=1).astype(np.uint8)
                mask_temp = Image.fromarray(mask_max*255)
                mask_max = np.zeros_like(mask_max)
                mask_temp = mask_temp.convert("L")
                # mask_temp = self._aug(mask_temp, transform1, self.mask_transform, state)  # [0, 1], for bg seg
                mask_temp = self._aug(mask_temp, self.mask_transform, None, state)
                input_masks.append(mask_temp)
                
            else:
                mask_max = mask_max + mask_temp
                mask_max = np.array(mask_max>=1).astype(np.uint8)
            '''
            mask_temp = Image.fromarray(mask_temp*255)
            mask_temp = mask_temp.convert("L")
            mask_temp = self._aug(mask_temp, self.mask_transform, None, state)
            input_masks.append(mask_temp)


        input_imgs = torch.stack(input_imgs, dim=0)  # T C H W
        input_masked_imgs = torch.stack(input_masked_imgs, dim=0)  # T C H W
        input_masks = torch.stack(input_masks, dim=0)

        outputs = {'target_images': input_imgs, 'input_masked_imgs': input_masked_imgs,
                   'input_masks': input_masks, 'caption': vid_caption, 'fps': fps
                #    'text_embeds': text_embeds
                   }

        return outputs
    