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
    def __init__(self,  data_loader, pairs_parser_factory, feature_extractor, distance_metric = 0):
        """Init Evaluator.

        Args:
            data_loader(object): a test data loader. 
            pairs_parser_factory(object): factory to produce the parser to parse test pairs list.
            feature_extractor(object): a feature extractor.
            distance_metric (int): type of distance metric (support for euclidean or cosine)
        """
        self.data_loader = data_loader
        pairs_parser = pairs_parser_factory.get_parser()
        self.pair_list = pairs_parser.parse_pairs()
        self.feature_extractor = feature_extractor
        self.distance_metric = distance_metric

    def eval(self, model):
        image_name2feature = self.feature_extractor.extract_online(model, self.data_loader)
        mean_dis_false, mean_dis_true, mean_acc, mean_tpr, mean_fpr ,std, best_thres = self.test_one_model(self.pair_list, image_name2feature)
        
        return mean_dis_false, mean_dis_true, mean_acc, mean_tpr, mean_fpr ,std, best_thres

    def distance(self, feat1, feat2, is_normalize=True):
        if is_normalize:
            feat1 = feat1 / np.linalg.norm(feat1)
            feat2 = feat2 / np.linalg.norm(feat2)
        if self.distance_metric == 0: 
            # euclidian distance
            diff = np.subtract(feat1,feat2)
            cur_score = math.sqrt(np.sum(np.square(diff),0))
        elif self.distance_metric==1: 
            # Distance based on cosine similarity
            cur_score = np.dot(feat1,feat2)
        else: 
            raise 'Undefined distance metric %d' % self.distance_metric 
        return cur_score


    def test_one_model(self, test_pair_list, image_name2feature, is_normalize = True):
        """Get the accuracy of a model.
        
        Args:
            test_pair_list(list): the pair list given by PairsParser. 
            image_name2feature(dict): the map of image name and it's feature.
            is_normalize(bool): wether the feature is normalized.

        Returns:
            mean_predict_label_false(float): mean of positive pair for theshold predict 
            mean_predict_labe_true(float):   mean of negative pair for theshold predict
            mean_acc(float): mean of accuracy 
            mean_tpr(float): mean of true positive rate 
            mean_fpr(float): mean of false positive rate
            std(float):      Standard deviation
            best_thres(float): best of theshold for metric true positive rate - false positive rate 

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

        mean_predict_label_false = np.mean(train_score_list[train_label_list==1])
        mean_predict_labe_true  = np.mean(train_score_list[train_label_list==0])                           

        accu_list = []
        tpr_list = [] 
        fpr_list = []
        best_thres = self.getThreshold(train_score_list, train_label_list)
        positive_score_list = train_score_list[train_label_list == 1]
        negtive_score_list  = train_score_list[train_label_list == 0]

        if not self.distance_metric:       
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
        
        else: 

            true_pos_pairs = np.sum(positive_score_list > best_thres)
            true_neg_pairs = np.sum(negtive_score_list < best_thres)
            false_neg_pairs = np.sum(positive_score_list < best_thres) 
            false_pos_pairs = np.sum(negtive_score_list > best_thres) 
        
            print('TP:{} TN:{} FN:{} FP:{}'.format(true_pos_pairs,true_neg_pairs,false_neg_pairs, false_pos_pairs))
            tpr_list.append( true_pos_pairs/(true_pos_pairs + false_neg_pairs) )
            fpr_list.append(false_pos_pairs/(false_pos_pairs + true_neg_pairs))
            accu_list.append((true_pos_pairs + true_neg_pairs) / train_score_list.shape[0])
        
            mean_acc = np.mean(accu_list)
            mean_tpr = np.mean(tpr_list)
            mean_fpr = np.mean(fpr_list)
        
        std = np.std(accu_list, ddof=1) / np.sqrt(10) #ddof=1, division 9.
        
        return mean_predict_label_false, mean_predict_labe_true, mean_acc, mean_tpr, mean_fpr ,std, best_thres

    def getThreshold(self, score_list, label_list):
        """Get the best threshold by train_score_list and train_label_list.
        Args:
            score_list(ndarray): the score list of all pairs.
            label_list(ndarray): the label list of all pairs.
            
        The function support for two of type caculate metric simility . 
        if distance metric = 0. That mean caculate  Euclidean distance and by define range of threshold in range (begin = 0.5, end = 1.6, 0.1 step). 
        and if distance metric = 1. That mean caculate cosine similaty.
        Returns:
            best_thres(float): the best threshold that computed by train set.
        """
        pos_score_list = score_list[label_list == 1]
        neg_score_list = score_list[label_list == 0]
        pos_pair_nums = pos_score_list.size
        neg_pair_nums = neg_score_list.size

        if self.distance_metric == 0: 
            fpr_list = []
            tpr_list = []
            threshold_list = np.arange(0.5, 1.6, 0.1)
            for threshold in threshold_list:
                fpr = np.sum(neg_score_list < threshold) / neg_pair_nums
                tpr = np.sum(pos_score_list < threshold) /pos_pair_nums

                fpr_list.append(fpr)
                tpr_list.append(tpr)
            fpr = np.array(fpr_list)
            tpr = np.array(tpr_list)
            best_index = np.argmax(tpr - fpr)
            best_thres = threshold_list[best_index]
        else: 
            num_thresholds = 1000
            score_max = np.max(score_list)
            score_min = np.min(score_list)
            score_span = score_max - score_min
            step = score_span / num_thresholds
            threshold_list = score_min +  step * np.array(range(1, num_thresholds + 1)) 
            fpr_list = []
            tpr_list = []
            for threshold in threshold_list:
                fpr = np.sum(neg_score_list > threshold) / neg_pair_nums
                tpr = np.sum(pos_score_list > threshold) /pos_pair_nums
                fpr_list.append(fpr)
                tpr_list.append(tpr)
            fpr = np.array(fpr_list)
            tpr = np.array(tpr_list)
            best_index = np.argmax(tpr-fpr)
            best_thres = threshold_list[best_index]
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
        best_thres = threshold_list[best_index]

        return  best_thres