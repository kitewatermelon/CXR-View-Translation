import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import UnidentifiedImageError

class Pix2PixMedicalImageDataset(Dataset):
    def __init__(self, df, transform=None, mode='P2L', max_samples=100, option=None, p_no=(10,19)):
        """
        Args:
            df (pd.DataFrame): DataFrame with 'subject_id', 'study_id', 'ViewPosition', 'path', and 'label' columns.
            transform (callable, optional): Optional transform to be applied on a sample.
            mode (str, optional) : Available modes are 'P2L', 'L2P'
            max_samples (int, optional): Maximum number of samples to be used in the dataset. Defaults to 100.
            p_no : Directory range and the number ex) (a=10, b=19) p10 to p19 data used. (min = 10 max = 19)
        """ 
        self.transform = transform
        self.mode = mode
        self.max_samples = max_samples
        self.option = option
        self.p_no = p_no
        self.df = df
        self.df = self._set_data()
        self.paired_df = self._create_pairs(self.df)
        self.paired_df = self.paired_df.head(self.max_samples)
        print("[INFO] dataset init done.")

    def _create_pairs(self, df):
        paired_data = []
        print("[INFO] making pairs of LAT and PA...")

        for _, group in df.groupby(['subject_id', 'study_id']):
            if set(group['ViewPosition']) >= {'PA', 'LATERAL'}:
                pa_image_path = group[group['ViewPosition'] == 'PA']['path'].values[0]
                lat_image_path = group[group['ViewPosition'] == 'LATERAL']['path'].values[0]
                label = group['label'].values[0] 
                paired_data.append({'subject_id': group['subject_id'].values[0],
                                    'study_id': group['study_id'].values[0],
                                    'pa_image_path': pa_image_path,
                                    'lat_image_path': lat_image_path,
                                    'label': label})

        print("[INFO] making pairs has been done.")

        return pd.DataFrame(paired_data)

    def _set_data(self):
        start, end = self.p_no
        regex = '|'.join([f"/p{i}/" for i in range(start, end + 1)])
        
        # 디렉토리와 라벨 조건으로 필터링
        filtered_df = self.df[
            self.df['path'].str.contains(regex) & (self.df['label'] == self.option)
        ]
        
        print(f"[INFO] Filtered dataset: {len(filtered_df)} entries with label '{self.option}'.")
        return filtered_df

    def __len__(self):
        return len(self.paired_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample = self.paired_df.iloc[idx]
        pa_image_path = sample['pa_image_path']
        lat_image_path = sample['lat_image_path']
        label = sample['label']
        
        if self.mode == 'P2L':
            input_image = self.load_image(pa_image_path)
            target_image = self.load_image(lat_image_path)
        
        elif self.mode == 'L2P':
            input_image = self.load_image(lat_image_path)
            target_image = self.load_image(pa_image_path)
            
        if self.transform:
            input_image = self.transform[0](input_image)
            target_image = self.transform[1](target_image)
        
        return {'input': input_image, 'target': target_image}


    def load_image(self, image_path):
        try:
            with open(image_path, 'rb') as img_file:
                img_data = img_file.read()
            img = Image.open(BytesIO(img_data)).convert('RGB')
            return img
        except (FileNotFoundError, UnidentifiedImageError) as e:
            print(f"Failed to load image from path: {image_path} - {e}")
            return None  # 실패 시 None 반환

    
    def show_sample(self, idx):
        sample = self.__getitem__(idx)
        input_image = sample['input'].numpy().transpose(1, 2, 0)
        target_image = sample['target'].numpy().transpose(1, 2, 0)
        
        print(f"[INFO] Input (PA) image: {self.paired_df.iloc[idx]['pa_image_path']}")
        print(f"[INFO] Target (LATERAL) image: {self.paired_df.iloc[idx]['lat_image_path']}")
        print(f"[INFO] Label : {sample['label']}")
        
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(input_image, cmap='gray')
        axes[0].set_title("Input (PA)")
        axes[0].axis('off')
        
        axes[1].imshow(target_image, cmap='gray')
        axes[1].set_title("Target (LATERAL)")
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    import os
    import data_prepareing

    print("🚀 Starting Pix2Pix Medical Image Dataset preparation...")
    try:
        if os.path.exists("processed.csv"):
            print("✅ Processed dataset found. Loading...")
        else:
            raise FileNotFoundError("❌ Processed dataset not found.")
    except FileNotFoundError:
        print("⚙️ Initiating dataset preparation...")
        data_prepareing.get_data()
        data_prepareing.get_data()
        print("✅ Dataset preparation complete.")
    
    print("📂 Loading dataset...")
    df = pd.read_csv("processed.csv")
    print(f"✅ Loaded {len(df)} entries from 'processed.csv'.")

    dataset = Pix2PixMedicalImageDataset(
        df, 
        transform=None, 
        mode='P2L', 
        max_samples=len(df), 
        option="No Finding",
        p_no=(10,19)
    )
    print(f"✅ Dataset ready for use. \n length : {len(dataset)} ")
