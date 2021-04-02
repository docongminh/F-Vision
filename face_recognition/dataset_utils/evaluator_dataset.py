"""
@author: Haoran Jiang, Jun Wang
@date: 20201013
@contact: jun21wangustc@gmail.com
"""

import os
import sys
import numpy as np
import math 
class Evaluator(object):
    """Implementation of LFW test protocal.
    
    Attributes:
        data_loader(object): a test data loader.
        pair_list(list): the pair list given by PairsParser.
        feature_extractor(object): a feature extractor.
    """
    def __init__(self, data_loader, pairs_parser_factory, feature_extractor):
        """Init Evaluator.

        Args:
            data_loader(object): a test data loader. 
            pairs_parser_factory(object): factory to produce the parser to parse test pairs list.
            pair_list(list): the pair list given by PairsParser.
            feature_extractor(object): a feature extractor.
        """
        self.data_loader = data_loader
        pairs_parser = pairs_parser_factory.get_parser()
        self.pair_list = pairs_parser.parse_pairs()
        self.feature_extractor = feature_extractor

    def eval(self, model):
        image_name2feature = self.feature_extractor.extract_online(model, self.data_loader)
        mean_dis_false, mean_dis_true, mean_acc, mean_tpr, mean_fpr ,std, best_thres = self.test_one_model(self.pair_list, image_name2feature)
        
        return mean_dis_false, mean_dis_true, mean_acc, mean_tpr, mean_fpr ,std, best_thres

    def distance(self, feat1, feat2, is_normalize=True):
        if is_normalize:
            feat1 = feat1 / np.linalg.norm(feat1)
            feat2 = feat2 / np.linalg.norm(feat2)
        diff = np.subtract(feat1,feat2)
        dist = np.sum(np.square(diff),0)

        return math.sqrt(dist)

    def test_one_model(self, test_pair_list, image_name2feature, is_normalize = True):
        """Get the accuracy of a model.
        
        Args:
            test_pair_list(list): the pair list given by PairsParser. 
            image_name2feature(dict): the map of image name and it's feature.
            is_normalize(bool): wether the feature is normalized.

        Returns:
            mean: estimated mean accuracy.
            std: standard error of the mean.
        """
       
        size = len(test_pair_list) 
        subsets_score_list = np.zeros((size), dtype = np.float32)
        subsets_label_list = np.zeros((size), dtype = np.int8)

        for index, cur_pair in enumerate(test_pair_list):
            image_name1 = cur_pair[0]
            image_name2 = cur_pair[1]
            label = cur_pair[2]

            feat1 = image_name2feature[image_name1]
            feat2 = image_name2feature[image_name2]

            subsets_label_list[index] = label
            subsets_score_list[index] = self.distance(feat1, feat2)

        train_score_list =  subsets_score_list
        train_label_list =  subsets_label_list

        mean_dis_false = np.mean(train_score_list[train_label_list==1])
        mean_dis_true  = np.mean(train_score_list[train_label_list==0])                           

        accu_list = []
        tpr_list = [] 
        fpr_list = []

        threshold_list = np.arange(0.5, 1.6, 0.1)
        best_thres = self.getThreshold(train_score_list, train_label_list, threshold_list)

        positive_score_list = train_score_list[train_label_list == 1]
        negtive_score_list  = train_score_list[train_label_list == 0]
        
        true_pos_pairs = np.sum(positive_score_list < best_thres)
        true_neg_pairs = np.sum(negtive_score_list > best_thres)
        false_neg_pairs = np.sum(positive_score_list > best_thres) 
        false_pos_pairs = np.sum(negtive_score_list < best_thres)  

        print('TP:{} TN:{} FN:{} FP:{}'.format(true_pos_pairs,true_neg_pairs,false_neg_pairs, false_pos_pairs))
        tpr_list.append( true_pos_pairs/(np.sum(positive_score_list)))
        fpr_list.append( false_pos_pairs/(np.sum(negtive_score_list)))
        accu_list.append((true_pos_pairs + true_neg_pairs)/train_score_list.shape[0])

        mean_acc = np.mean(accu_list)
        mean_tpr = np.mean(tpr_list)
        mean_fpr = np.mean(fpr_list)
        std = np.std(accu_list, ddof=1) / np.sqrt(10) #ddof=1, division 9.
        
        return mean_dis_false, mean_dis_true, mean_acc, mean_tpr, mean_fpr ,std, best_thres

    def getThreshold(self, score_list, label_list, threshold_list):
        """Get the best threshold by train_score_list and train_label_list.
        Args:
            score_list(ndarray): the score list of all pairs.
            label_list(ndarray): the label list of all pairs.
            num_thresholds(int): the number of threshold that used to compute roc.
        Returns:
            best_thres(float): the best threshold that computed by train set.
        """
        pos_score_list = score_list[label_list == 1]
        neg_score_list = score_list[label_list == 0]
        pos_pair_nums = pos_score_list.size
        neg_pair_nums = neg_score_list.size

        fpr_list = []
        tpr_list = []
        acc_list = []
        for threshold in threshold_list:
            fpr = np.sum(neg_score_list < threshold) / neg_pair_nums
            tpr = np.sum(pos_score_list < threshold) /pos_pair_nums

            true_pos_pairs = np.sum(pos_score_list < threshold)
            true_neg_pairs = np.sum(neg_score_list > threshold)
            acc_list.append((true_pos_pairs+true_neg_pairs)/score_list.shape[0] )

            fpr_list.append(fpr)
            tpr_list.append(tpr)
        fpr = np.array(fpr_list)
        tpr = np.array(tpr_list)
        best_index = np.argmax(tpr - fpr)
        # best_index = np.argmax(np.array(acc_list)) 
        best_thres = threshold_list[best_index]
        """
            TPR: Độ nhạy model còn được gọi là TPR(True positive rate) cho 
            biết mức độ dự báo chính xác trong nhóm sự kiện positive.
            FPR: Cho biết mức độ dự báo sai một sự kiện khi nó là negative nhưng kết luận là positive.
            TPR - FPR:  toi thieu rui ro du doan sai. max TP va TN
        """
        print ('best threshold', best_thres)
        
        return  best_thres