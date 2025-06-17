# coding: utf-8
# @email: enoche.chow@gmail.com

"""
Main entry
# UPDATED: 2022-Feb-15
##########################
"""

import os
import argparse
from utils.quick_start import quick_start
os.environ['NUMEXPR_MAX_THREADS'] = '48'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='VBPR', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='beauty', help='name of datasets')

    config_dict = {
        #'dropout': [0.2],
        'reg_weight': [2.0],
        #'learning_rate': [0.0005],
        #'reg_weight': [0.0001,0.00001],
        #'mm_image_weight' : [0.1],
        #'knn_k': [5],
        #'k':[40],
        #'n_mm_layers': [1],
        #'u_layers': [1],
        #'ssl_temp': [1.0],
        #'ssl_alpha':[0.1],
        #'reg_weight': [1e-04],
        'gpu_id': 0,
    }

    args, _ = parser.parse_known_args()

    quick_start(model=args.model, dataset=args.dataset, config_dict=config_dict, save_model=True)


