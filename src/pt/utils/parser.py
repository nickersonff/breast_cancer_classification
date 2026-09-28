from abc import ABC, abstractmethod
import glob
import os
import pandas as pd
from src.pt.utils.constants import Constants

# --- Classe Base (Interface) ---
class BaseDatasetParser(ABC):
    def __init__(self, metadata_file: str = None):
        self.config = Constants.get_config()
        self.metadata_file = metadata_file

    @abstractmethod
    def parse(self):
        """
        Deve retornar duas listas: (file_paths, labels)
        """
        pass

    def birads_cleaner(self, df, column_name='breast_birads'):
        group_birads = self.config['dataset_config'].get('group_birads', True)
        df_final = df[df[column_name].isin([1, 2, 4, 5, 6])] if group_birads else df[df[column_name].isin([1, 2, 3, 4, 5, 6])]
        return df_final
    
    def make_grouping_birads(self, b):
        dic_birads_grouping = {
            1: 0,  # BI-RADS 1 -> Categoria 0
            2: 0,  # BI-RADS 2 -> Categoria 0
            3: 0,  # BI-RADS 3 -> Categoria 0
            4: 1,  # BI-RADS 4 -> Categoria 1
            5: 1,   # BI-RADS 5 -> Categoria 1
            6: 1   # BI-RADS 6 -> Categoria 1
        }

        return dic_birads_grouping[b]

# --- Parser para o VinDr-Mammo (geralmente usa CSV de metadados) ---
class VinDrParser(BaseDatasetParser):
    def parse(self):
        columns = {
            'study_id': str,
            'image_id': str,
            'manufacturer': str,
            'breast_birads': int,
            'view_position': str,
            'laterality': str,
        }
        df = pd.read_csv(self.metadata_file, dtype=columns)
        df = self.birads_cleaner(df=df, column_name='breast_birads')
        manufacturers = self.config['dataset_config'].get('manufacturers', [])
        if len(manufacturers) > 0:
            df = df[df['manufacturer'].isin(manufacturers)]

        data_list = []
        group_birads = self.config['dataset_config'].get('group_birads', True)

        for _, row in df.iterrows():
            file_name = f"{row['study_id']}_{row['image_id']}.npy"
            dicom_file = f"{row['study_id']}/{row['image_id']}.dicom"
            dicom_root = self.config['io_dirs'].get('dicom_root_VINDR')
            sample_info = {
                'npy': file_name,
                'dicom': os.path.join(dicom_root, dicom_file),
                'label': self.make_grouping_birads((row['breast_birads'])) if group_birads else (row['breast_birads']-1),
                'patient_id': row.get('study_id', None),
                'view': row.get('view_position', None),
                'laterality': row.get('laterality', None),
                'dataset': 'vindr'
            }
            data_list.append(sample_info)
        print(f'Number of samples in VinDr dataset: {len(data_list)}')     
        return data_list

# --- Parser para o CBIS-DDSM (estruturado por pastas/pacientes) ---
class CBISDDSMParser(BaseDatasetParser):
    def parse(self):
        columns = {
                    'patient_id': str,
                    'image file path': str,
                    'assessment': int,
                    'image view': str,
                    'left or right breast': str,
                }
        df = pd.read_csv(self.metadata_file, dtype=columns)
        df = df.groupby(['patient_id', 'image view', 'left or right breast', 'image file path'])['assessment'].max().reset_index()
        df = self.birads_cleaner(df, column_name='assessment')
        data_list = []
        group_birads = self.config['dataset_config'].get('group_birads', True)
        for _, row in df.iterrows():
            dicom_root = self.config['io_dirs'].get('dicom_root_DDSM')
            dir_name = row['image file path'].split('/')[0]
            file_name = f"{dir_name}.npy"
            dicom_file = glob.glob(os.path.join(dicom_root, dir_name, "**", "*.dcm"), recursive=True)
            sample_info = {
                'npy': file_name,
                'dicom': dicom_file[0],
                'label': self.make_grouping_birads(row['assessment']) if group_birads else (row['assessment']-1),
                'patient_id': row.get('patient_id', None),
                'view': row.get('image view', None),
                'laterality': row.get('left or right breast', None),
                'dataset': 'cbis-ddsm'
            }
            data_list.append(sample_info)

        print(f'Number of samples in CBIS-DDSM dataset: {len(data_list)}')
        return data_list