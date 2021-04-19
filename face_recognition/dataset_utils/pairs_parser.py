""" 
@author: Jixuan Xu, Jun Wang
@date: 20201015
@contact: jun21wangustc@gmail.com
""" 
import os 
import scipy.io as scio
from abc import ABCMeta, abstractmethod


class PairsParser(metaclass=ABCMeta):
    """Parse the pair list for lfw based protocol.
    Because the official pair list for different dataset(lfw, cplfw, calfw ...) is different, 
    we need different method to parse the pair list of different dataset.
    
    Attributes:
        pairs_file(str): the path of the pairs file that was released by official.
    """
    def __init__(self, pairs_file):
        """Init PairsParser
            
        Args:
            pairs_file(str): the path of the pairs file that was released by official.
        """
        self.pairs_file = pairs_file
    def parse_pairs(self):
        """The method for parsing pair list.
        """
        pass

class LFW_PairsParser(PairsParser):
    """The pairs parser for lfw.
    """
    def parse_pairs(self):
        cnt = 0
        test_pair_list = []
        pairs_file_buf = open(self.pairs_file)
        line = pairs_file_buf.readline() # skip first line
        line = pairs_file_buf.readline().strip()
        while line:
            line_strs = line.split('\t')
            if len(line_strs) == 3:
                person_name = line_strs[0]
                image_index1 = line_strs[1]
                image_index2 = line_strs[2]
                image_name1 = person_name + '/' + person_name + '_' + image_index1.zfill(4) + '.jpg'
                image_name2 = person_name + '/' + person_name + '_' + image_index2.zfill(4) + '.jpg'
                label = 1
            elif len(line_strs) == 4:
                person_name1 = line_strs[0]
                image_index1 = line_strs[1]
                person_name2 = line_strs[2]
                image_index2 = line_strs[3]
                image_name1 = person_name1 + '/' + person_name1 + '_' + image_index1.zfill(4) + '.jpg'
                image_name2 = person_name2 + '/' + person_name2 + '_' + image_index2.zfill(4) + '.jpg'
                label = 0
            else:
                raise Exception('Line error: %s.' % line)
            test_pair_list.append((image_name1, image_name2, label))
            line = pairs_file_buf.readline().strip()

        return test_pair_list

class VN_Celeb_PairsParser(PairsParser):
    """The pairs parser for VN celeb dataset.
    returns: 
            list of tupble (img_first, img_second, labels of pair)
    """
    def parse_pairs(self):        
        pair_list = []
        pairs_file_buf = open(self.pairs_file)
        line1 = pairs_file_buf.readline().strip()
        # print(line1)
        while line1:
            line2 = pairs_file_buf.readline().strip()
            image_name1 = line1.split(' ')[0]
            image_name2 = line1.split(' ')[1]
            label = line1.split(' ')[2]
            pair_list.append((image_name1, image_name2, int(label)))
            line1 = line2

        
        size = len(pair_list)//2
        test_pair_list = []
        positive_start = 0   # 0-> size - 1
        negtive_start = size # size -> size*2

        nofsplit = 10
        k_fold = size// nofsplit 
        for set_idx in range(k_fold):
            positive_index = positive_start + nofsplit * set_idx
            negtive_index = negtive_start + nofsplit * set_idx
            cur_positive_pair_list = pair_list[positive_index : positive_index + nofsplit]
            cur_negtive_pair_list = pair_list[negtive_index : negtive_index + nofsplit]
            test_pair_list.extend(cur_positive_pair_list)
            test_pair_list.extend(cur_negtive_pair_list)

        return test_pair_list

     

class PairsParserFactory(object):
    """The factory used to produce different pairs parser for different dataset.

    Attributes:
        pairs_file(str): the path of the pairs file that was released by official.
    """
    def __init__(self, pairs_file):
        self.pairs_file = pairs_file
    def get_parser(self):
        pairs_parser =  VN_Celeb_PairsParser(self.pairs_file)
        # pairs_parser = LFW_PairsParser(self.pairs_file)

        return pairs_parser
