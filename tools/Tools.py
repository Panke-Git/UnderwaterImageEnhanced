"""
    @Project: UnderwaterImageEnhanced
    @Author: Panke
    @FileName: Tools.py
    @Time: 2025/5/23 23:30
    @Email: None
"""
from src.utils.train_utils import generate_experiment_id
from src.utils.record_utils import record_model_description

# expt_id = generate_experiment_id(model='Unet',
#                                  dataset='LSUI',
#                                  loss='L1SSIM',
#                                  note='SmoothL1Loss')
# print(expt_id)

json_path = r'E:\PythonProject\01_Personal\UnderwaterImageEnhanced\src\models\01_ModelDescription.json'

record_model_description(model_description='测试的而已', model_name='Unet', json_file=json_path)
