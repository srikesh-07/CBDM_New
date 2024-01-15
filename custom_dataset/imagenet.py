from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
import os
import random
from PIL import Image


image_size = 64
# Data transformation with augmentation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# Dataset
class LT_Dataset(Dataset):
    
    def __init__(self, root, txt, transform=None, random_sampling=None, random_seed=42):
        self.img_path = []
        self.targets = []
        self.transform = transform
        self.random_sampling = random_sampling
        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.targets.append(int(line.split()[1]))
        if random_sampling:
            random.seed(random_seed)
            self.indices = random.sample(range(0, len(self.targets)), k=random_sampling)
            
        
    def __len__(self):
        if self.random_sampling:
            return len(self.indices)
        else:
            return len(self.targets)
        
    def __getitem__(self, index):

        if self.random_sampling:
            index = self.indices[index]

        path = self.img_path[index]
        label = self.targets[index]
        
        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        
        if self.transform is not None:
            sample = self.transform(sample)

        # return sample, label, path
        return sample, label


# Load datasets
def load_data(data_root, dist_path, phase="train", transform=None, random_sampling=None, random_seed=42):
    
    dataset = os.path.basename(dist_path)
    txt = '%s/%s_%s.txt'%(dist_path, dataset, (phase if phase != 'train_plain' else 'train'))

    # print('Loading data from %s' % (txt))

    # if phase not in ['train', 'val']:
    #     transform = data_transforms['test']
    # else:
    #     transform = data_transforms[phase]

    # print('Use data transformation:', transform)

    dataset = LT_Dataset(data_root, txt, transform, random_sampling=random_sampling, random_seed=random_seed)

    return dataset, None, None

    # if phase == 'test' and test_open:
    #     open_txt = './data/%s/%s_open.txt'%(dataset, dataset)
    #     print('Testing with opensets from %s'%(open_txt))
    #     open_set_ = LT_Dataset('./data/%s/%s_open'%(dataset, dataset), open_txt, transform)
    #     set_ = ConcatDataset([set_, open_set_])

    # if sampler_dic and phase == 'train':
    #     print('Using sampler.')
    #     print('Sample %s samples per-class.' % sampler_dic['num_samples_cls'])
    #     return DataLoader(dataset=set_, batch_size=batch_size, shuffle=False,
    #                        sampler=sampler_dic['sampler'](set_, sampler_dic['num_samples_cls']),
    #                        num_workers=num_workers)
    # else:
    #     print('No sampler.')
    #     print('Shuffle is %s.' % (shuffle))
    #     return DataLoader(dataset=set_, batch_size=batch_size,
    #                       shuffle=shuffle, num_workers=num_workers)
        
    
    
