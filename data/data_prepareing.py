import os
import re
import pandas as pd
from utils.utils import PATH


def get_data():
    try:
        # 파일 존재 여부 확인
        if os.path.exists(PATH + "processed.csv"):
            print("[INFO] Processed data already exists. Skipping data processing.")
            df = pd.read_csv(PATH + "processed.csv")
            return df
        else:
            raise FileNotFoundError("File not found")  # 파일이 없으면 예외 발생
    except FileNotFoundError:
        print("[INFO] Processed data file not found. Starting data processing...")
        def extract_ids(image_path):
            match = re.search(r'files/p(\d{2})/p(\d+)/s(\d+)/([a-f0-9-]+)', image_path)
            if match:
                subject_id = match.group(2)
                study_id = match.group(3)
                dicom_id = match.group(4).split(".")[0]
                return dicom_id, subject_id, study_id
            return None, None, None

        def create_labels(row):
            labels = [col for col in merged_df.columns[6:] if row[col] == '1.0']
            return ', '.join(labels)

        def select_one_pa_lat(group):
            pa_row = group[group['ViewPosition'] == 'PA'].sample(n=1)
            lat_row = group[group['ViewPosition'] == 'LATERAL'].sample(n=1)
            return pd.concat([pa_row, lat_row])


        print("[INFO] Loading datasets...")
        negbio = pd.read_csv(PATH + "mimic-cxr-2.0.0-negbio.csv.gz", compression="gzip")
        metadata = pd.read_csv(PATH + "mimic-cxr-2.0.0-metadata.csv.gz", compression="gzip")
        
        print("[INFO] Converting data types...")
        negbio = negbio.astype(str)
        metadata = metadata.astype(str)
        print("[INFO] Reading image paths...")
        IMAGE_FILENAMES = PATH + "IMAGE_FILENAMES"
        with open(IMAGE_FILENAMES, 'r') as file:
            image_paths = [PATH + line.strip() for line in file]

        print("[INFO] Extracting IDs from image paths...")
        data = []
        for path in image_paths:
            dicom_id, subject_id, study_id = extract_ids(path)
            data.append([str(dicom_id), str(subject_id), str(study_id), path])

        path_df = pd.DataFrame(data, columns=["dicom_id", "subject_id", "study_id", "path"])

        print("[INFO] Merging datasets...")
        merged_df = pd.merge(
            path_df,
            metadata[['PerformedProcedureStepDescription', 'ViewPosition', 'subject_id', 'study_id', 'dicom_id']],
            on=['subject_id', 'study_id', 'dicom_id'],
            how='inner'
        )
        merged_df = pd.merge(merged_df, negbio, on=['subject_id', 'study_id'], how='inner')

        print("[INFO] Cleaning data...")
        merged_df = merged_df.fillna(0)
        merged_df = merged_df.drop('Support Devices', axis=1)


        merged_df['label'] = merged_df.apply(create_labels, axis=1)
        merged_df = merged_df.dropna()
        merged_df = merged_df.drop(
            merged_df[['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 
                       'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 
                       'Lung Opacity', 'No Finding', 'Pleural Effusion', 'Pleural Other', 'Pneumonia', 'Pneumothorax']],
            axis=1
        )

        print("[INFO] Filtering studies with both PA and LATERAL views...")
        PA_LAT_set_df = merged_df.groupby(['subject_id', 'study_id']).filter(lambda x: set(x['ViewPosition']) >= {'PA', 'LATERAL'})

        # 각 study_id 별로 PA와 LAT 각각 하나씩 선택
        print("[INFO] Selecting one PA and one LATERAL view per study...")

        df = PA_LAT_set_df.groupby(['subject_id', 'study_id']).apply(select_one_pa_lat).reset_index(drop=True)

        print("[INFO] Saving processed dataset to 'processed.csv'...")
        df.to_csv(PATH + "processed.csv", index=False)
        print("[INFO] Dataset processing complete. Saved to 'processed.csv'.")
        return df

if __name__ == "__main__":
    get_data()