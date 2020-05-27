import time
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import pandas_datareader as pdr

CODE = 'UDOW'
START_DATE = '2020-01-01'
TRAIN_RATIO = 0.8

# 야후 금융에서 조회하여 train, test로 나눔
def dataset_loader(stock_name, start, train_ratio):
  dataset = pdr.DataReader(stock_name, data_source="yahoo", start=start)
  date_split = str(dataset.index[int(train_ratio*len(dataset))]).split(' ')[0]

  return dataset[:date_split], dataset[date_split:], date_split
  
(train, test, date_split) = dataset_loader(CODE, START_DATE, TRAIN_RATIO)

print("test from ", date_split)
