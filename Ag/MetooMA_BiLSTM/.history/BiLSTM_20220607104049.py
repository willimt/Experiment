from os import sep
import time
import torch
import math
from d2l import torch as d2l
from torch import nn
from MyKu import processing
import nltk
import pandas as pd
from tqdm import tqdm
import twint

c = twint.Config()

c.Search = "ETH"
c.P
twint.run.Search(c)
