"""Helper for evaluation on the Labeled Faces in the Wild dataset 
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
from sklearn.model_selection import KFold
from scipy import interpolate
import math
import sys 
sys.path.append('../face_recognition/dataset_utils')

from verifacation import evaluate
from pdb import set_trace as bp
import torch
import torch.nn.functional as F
from train_dataset import NormBatch

from datetime import datetime
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
from PIL import Image
import bcolz
import io
import os



def gen_plot(fpr, tpr):
    """Create a pyplot plot and save to buffer."""
    plt.figure()
    plt.xlabel("FPR", fontsize = 14)
    plt.ylabel("TPR", fontsize = 14)
    plt.title("ROC Curve", fontsize = 14)
    plot = plt.plot(fpr, tpr, linewidth = 2)
    buf = io.BytesIO()
    plt.savefig(buf, format = 'jpeg')
    buf.seek(0)
    plt.close()

    return buf

def perform_val(model, conf, carray, issame, nrof_folds = 10, tta = False):
    """
    tta: normalize image array to same standard data training  
    """
    normbatch = NormBatch()
    model.eval() # switch to evaluation mode

    idx = 0
    embeddings = np.zeros([len(carray), conf.feat_dim])
    with torch.no_grad():
        while idx + conf.batch_size <= len(carray):
            batch = torch.tensor(carray[idx:idx + conf.batch_size][:, [2, 1, 0], :, :])
            if tta:
                fliped = normbatch.hflip_batch(batch)
                emb_batch = model.feed_emb(batch.to(conf.device)).cpu() + model.feed_emb(fliped.to(conf.device)).cpu()
                embeddings[idx:idx + conf.batch_size] = normbatch.l2_norm(emb_batch)
            else:
                embeddings[idx:idx + conf.batch_size] = model.feed_emb(batch.to(conf.device)).cpu()
            idx += conf.batch_size
        if idx < len(carray):
            batch = torch.tensor(carray[idx:])
            if tta:
                fliped = normbatch.hflip_batch(batch)
                emb_batch = model.feed_emb(batch.to(conf.device)).cpu() + model.feed_emb(fliped.to(conf.device)).cpu()
                embeddings[idx:] = normbatch.l2_norm(emb_batch)
            else:
                embeddings[idx:] = model.feed_emb(batch.to(conf.device)).cpu()
    tpr, fpr, accuracy, best_thresholds = evaluate(embeddings, issame, nrof_folds)
    buf = gen_plot(fpr, tpr)
    roc_curve = Image.open(buf)
    roc_curve_tensor = transforms.ToTensor()(roc_curve)

    return accuracy.mean(), best_thresholds.mean(), roc_curve_tensor