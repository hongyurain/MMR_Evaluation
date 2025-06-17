#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 22/11/17

@author: Maurizio Ferrari Dacrema
@author: Simone Boglio

"""

import argparse
from Recommender_import_list import *

from ParameterTuning.run_parameter_search import runParameterSearch_Collaborative
from ParameterTuning.SearchSingleCase import SearchSingleCase
from ParameterTuning.SearchAbstractClass import SearchInputRecommenderArgs



from Utils.ResultFolderLoader import ResultFolderLoader, generate_latex_hyperparameters
from Utils.assertions_on_data_for_experiments import assert_implicit_data, assert_disjoint_matrices
from Utils.plot_popularity import plot_popularity_bias, save_popularity_statistics

from functools import partial
import numpy as np
import os, traceback, argparse

from DataReader import DataReader


def read_data_split_and_search(dataset_name, datafolder, datapath, cold_start = False, cold_items=None,
                                          flag_baselines_tune = False):

    result_folder_path = datapath + datafolder + f'/traditional_single_domain_0523_{dataset_name}/'

    assert (cold_start is not True)
    dataset = DataReader(result_folder_path, dataset_name, datapath + datafolder + '/', type="original")
    URM_train = dataset.URM_DICT["URM_train"].copy()
    URM_validation = dataset.URM_DICT["URM_validation"].copy()
    URM_test = dataset.URM_DICT["URM_test"].copy()
    URM_test_negative = dataset.URM_DICT["URM_test_negative"].copy()

    # Ensure IMPLICIT data and DISJOINT sets
    # assert_implicit_data([URM_train, URM_validation, URM_test])
    print('URM_train = ', URM_train.shape)
    print('URM_validation = ', URM_validation.shape)
    print('URM_test = ', URM_test.shape)
    # assert_disjoint_matrices([URM_train, URM_validation, URM_test])

    # If directory does not exist, create
    if not os.path.exists(result_folder_path):
        os.makedirs(result_folder_path)

    metric_to_optimize = "NDCG"
    n_cases = 30
    n_random_starts = 15

    from Base.Evaluation.Evaluator import EvaluatorHoldout, EvaluatorNegativeItemSample

    if not cold_start:
        cutoff_list_validation = [10]
        cutoff_list_test = [1, 5, 10, 20]
    else:
        cutoff_list_validation = [20]
        cutoff_list_test = [20]

    evaluator_validation = EvaluatorHoldout(URM_validation, dataset_name_jing=dataset_name, dataset_is_valid_jing=True, cutoff_list=cutoff_list_validation)
    evaluator_test = EvaluatorHoldout(URM_test, dataset_name_jing=dataset_name, dataset_is_valid_jing=False, cutoff_list=cutoff_list_test)
    # print('URM_test = ',URM_test)
    # URM_test_list, URM_test_negative, cutoff_list, min_ratings_per_user=1, exclude_seen=True,
    #                  diversity_object = None,
    #                  ignore_items = None,
    #                  ignore_users = None
    # evaluator_validation = EvaluatorHoldout(URM_validation, dataset_name_jing=dataset_name, dataset_is_valid_jing=True, cutoff_list=cutoff_list_validation)
    # # evaluator_validation = EvaluatorNegativeItemSample(URM_validation, cutoff_list=cutoff_list_validation)
    # evaluator_test = EvaluatorNegativeItemSample(URM_test, URM_test_negative, cutoff_list=cutoff_list_test)

    ################################################################################################
    ###### KNN CF

    collaborative_algorithm_list = [
        Random,
        TopPop,
        UserKNNCFRecommender,
        ItemKNNCFRecommender,
        SLIM_BPR_Cython,
        MatrixFactorization_BPR_Cython,


        # P3alphaRecommender,
        # RP3betaRecommender,
        # PureSVDRecommender,
        # NMFRecommender,
        # IALSRecommender,
        # MatrixFactorization_FunkSVD_Cython,
        # EASE_R_Recommender,
        # SLIMElasticNetRecommender,
        ]

    # collaborative_algorithm_list = [
    #     MatrixFactorization_BPR_Cython,
    # ]

    runParameterSearch_Collaborative_partial = partial(runParameterSearch_Collaborative,
                                                       URM_train = URM_train,
                                                       URM_train_last_test = URM_train, # + URM_validation,
                                                       metric_to_optimize = metric_to_optimize,
                                                       evaluator_validation_earlystopping = evaluator_validation,
                                                       evaluator_validation = evaluator_validation,
                                                       evaluator_test = evaluator_test,
                                                       output_folder_path = result_folder_path,
                                                       parallelizeKNN = False,
                                                       allow_weighting = True,
                                                       resume_from_saved = True,
                                                       n_cases = n_cases,
                                                       n_random_starts = n_random_starts)

    if flag_baselines_tune:
        num_model_jing = 0
        for recommender_class in collaborative_algorithm_list:
            print('num_model_jing: {}'.format(num_model_jing))
            num_model_jing += 1
            try:
                runParameterSearch_Collaborative_partial(recommender_class)
            except Exception as e:
                print("On recommender {} Exception {}".format(recommender_class, str(e)))
                traceback.print_exc()


if __name__ == '__main__':
    dataset = 'douban1'
    # important: beauty art taobao
    # baby sports clothing

    """
    Can only change the data_path: datapath + datafolder + '/' + dataset_name + '_train.txt' 
    Same to validate.txt and test.txt
    All traditional results will be saved in: datapath + datafolder + '/' + 'traditional_single_domain' folder

    dataset format:
    four columns: user item rating timestamp
    """
    data_path = '../dataset/'
    data_folder = '{}'.format(dataset)
    dataset_name = '{}_tra'.format(dataset)

    data_folder = 'douban' # 'baby,sports,clothing'
    dataset_name = dataset


    # data path is datapath + datafolder + '/' + dataset_name + '_train.txt'


    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--baseline_tune',        help="Baseline hyperparameter search", type = bool, default = True)
    parser.add_argument('-a', '--DL_article_default',   help="Train the DL model with article hyperparameters", type = bool, default = True)
    parser.add_argument('-p', '--print_results',        help="Print results", type = bool, default = True)

    parser.add_argument('-t', '--DL_tune',              help="DL model hyperparameter search", type = bool, default = False)

    input_flags = parser.parse_args()
    print(input_flags)

    KNN_similarity_to_report_list = ["cosine", "dice", "jaccard", "asymmetric", "tversky"]

    from collections import namedtuple

    CustomRecommenderName = namedtuple('CustomRecommenderName', ['RECOMMENDER_NAME'])

    read_data_split_and_search(dataset_name, data_folder, data_path,
                                         cold_start = False,
                                         flag_baselines_tune=input_flags.baseline_tune
                                          )

    print('FINISH testing...')
