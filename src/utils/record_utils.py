"""
    @Project: UnderwaterImageEnhanced
    @Author: Panke
    @FileName: record_utils.py
    @Time: 2025/5/21 22:09
    @Email: None
"""

import json
import os

import pandas as pd

from src.utils.config import Config


def make_train_path(record_path, model_name, start_time):
    """
    创建本次训练需要的目录
    :param model_name: 模型描述，也可以认为是模型的名称，是某个模型的目录
    :param start_time: 训练时间，该模型本次训练的开始时间，为耳机目录
    :return: None
    """
    time_path = os.path.join(record_path, model_name, start_time)
    best_path = os.path.join(record_path, model_name, start_time, 'best_result')
    if not os.path.exists(time_path):
        os.makedirs(time_path)
    if not os.path.exists(best_path):
        os.makedirs(best_path)
    return time_path, best_path


def package_one_epoch(**kwargs):
    one_epoch_data = json.dumps(kwargs)
    return one_epoch_data


def save_train_data(target_path, start_time, end_time, list_data, top_data):
    # 保存数据到Excel文件
    records = [json.loads(item) for item in list_data]
    df = pd.DataFrame(records)
    excel_name = 'Trian_' + '_' + start_time + '——' + end_time + '.xlsx'

    if not os.path.exists(target_path):
        os.makedirs(target_path)
    excel_path = os.path.join(target_path, excel_name)
    df.to_excel(str(excel_path), index=False, sheet_name='Train_data')

    # 保存数据到JSON文件
    json_name = 'Trian_' + '_' + start_time + '——' + end_time + '.json'
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    json_path = os.path.join(target_path, json_name)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False, indent=4)
    f.close()

    # 保存最佳效果数据
    top_path = os.path.join(target_path, 'best_result', 'top_result.json')
    with open(top_path, 'w', encoding='utf-8') as f:
        json.dump(top_data, f, ensure_ascii=False, indent=4)
    f.close()
    return excel_path, json_path, top_path


def save_train_config(save_path, **kwargs):
    json_data = json.dumps(kwargs, indent=4, ensure_ascii=False)
    file_path = os.path.join(save_path, 'Train config.txt')
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(json_data)
    f.close()
    return file_path
