import os
import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


# We force the model to ignore color and orientation so it focuses on cell shape.
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=45),
    # ColorJitter randomly changes brightness, contrast, saturation, and hue.
    # This prevents the model from memorizing SIPaKMeD's specific lab stain!
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# For testing, we don't augment the images, we just resize and normalize them.
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class CervicalDataset(Dataset):
    def __init__(self, root_dir, dataset_type, transform=None):
        self.root_dir = root_dir
        self.dataset_type = dataset_type
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        self._load_data()

    def _load_data(self):
        all_files = glob.glob(os.path.join(self.root_dir, '**', '*.*'), recursive=True)
        valid_extensions = ('.bmp', '.jpg', '.jpeg', '.png')
        
        for file_path in all_files:
            if not file_path.lower().endswith(valid_extensions):
                continue
                
            folder_name = os.path.basename(os.path.dirname(file_path)).lower()
            label = self._map_to_binary(folder_name)
            
            if label is not None:
                self.image_paths.append(file_path)
                self.labels.append(label)

    def _map_to_binary(self, folder_name):
        if self.dataset_type == 'sipakmed':
            normal_classes = ['parabasal', 'superficial-intermediate', 'im_parabasal', 'im_superficial-intermediate']
            abnormal_classes = ['dyskeratotic', 'koilocytotic', 'metaplastic', 'im_dyskeratotic', 'im_koilocytotic']
            
            if any(c in folder_name for c in normal_classes): return 0
            if any(c in folder_name for c in abnormal_classes): return 1
            
        elif self.dataset_type == 'herlev':
            normal_classes = ['normal_superficial', 'normal_intermediate', 'normal_columnar', 'im_metaplastic']
            abnormal_classes = ['light_dysplastic', 'moderate_dysplastic', 'severe_dysplastic', 'carcinoma_in_situ']
            
            if any(c in folder_name for c in normal_classes): return 0
            if any(c in folder_name for c in abnormal_classes): return 1
            
        return None

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label


def get_dataloaders(sipakmed_path, herlev_path, batch_size=32):
    # Apply heavy augmentation to the source training data
    source_dataset = CervicalDataset(sipakmed_path, 'sipakmed', transform=train_transforms)
    # Target data is just for testing/adaptation, no crazy augmentation needed
    target_dataset = CervicalDataset(herlev_path, 'herlev', transform=test_transforms)
    
    source_loader = DataLoader(source_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    target_loader = DataLoader(target_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    return source_loader, target_loader
