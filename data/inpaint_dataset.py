import os
import glob
from PIL import Image
import numpy as np
import cv2
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class In_mouth_pic(Dataset):
    def __init__(self, args, train=False):
        #print(data_path, train_path, tid)
        

        if args.data_type=='smile':
            self.all_files = glob.glob('/mnt/share/shenfeihong/data/smile_ylc/*', recursive=False)[10:]
        else:
            data_path = '/mnt/share/shenfeihong/data/abrasion/in_mouth_pic'
            self.all_files = glob.glob(os.path.join(data_path,'*/*/*/*.jpg'), recursive=False)[10:]
        if not train:
            print('eval')
            self.all_files = glob.glob(os.path.join(data_path,'*/*/*/*.jpg'), recursive=False)[:10]
        self.transform = transforms.Compose(
                            [
                                transforms.RandomCrop(size=(args.image_size, args.image_size)),
                                transforms.ToTensor()
                            ]
                        )
        print('total image:', len(self.all_files))
    
    def __len__(self):
        return len(self.all_files)
    
    def __getitem__(self, index):
        frame_file = self.all_files[index]
        ske_file = frame_file.replace('frame','ske')
        
        frame = Image.open(frame_file)
        # ske = Image.open(ske_file)

        img = self.transform(frame)*2-1
        # cond = self.transform(ske)

        return {'images': img}

def get_loader(args, train=False):
    dataset = In_mouth_pic(args, train=train)
    loader = DataLoader(
        dataset=dataset,
        batch_size=args.train_batch_size,
        num_workers=4,
        pin_memory=True,
    )
    return loader

def b():
    loader = get_loader()
    data = next(iter(loader))
    print(data['image'].shape, data['cond'].shape)            
if __name__=="__main__":
    b()
