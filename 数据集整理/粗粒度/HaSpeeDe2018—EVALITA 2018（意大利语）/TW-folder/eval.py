# -*- coding: utf-8 -*-

"""
evaluation script for the HaSpeeDe 2018 shared task

USAGE: eval.py [reference] [predicted]
"""

import argparse, sys
from pandas_ml import ConfusionMatrix
from sklearn.metrics import precision_recall_fscore_support as score


def preproc(infile):
    y = []
    
    reader = infile.read().splitlines() 
    for row in reader:
        label = row.split('\t')[2]
        y.append(label)

    return y


def eval(y_test, y_predicted):    

    precision, recall, fscore, _ = score(y_test, y_predicted)
    print'\n     {0}   {1}'.format("0","1")
    print'P: {}'.format(precision)
    print'R: {}'.format(recall)
    print'F: {}'.format(fscore)

    mprecision, mrecall, mfscore, _ = score(y_test, y_predicted, average='macro')
    print'\n MACRO-AVG'
    print'P: {}'.format(mprecision)
    print'R: {}'.format(mrecall)
    print'F: {}'.format(mfscore)
                        
    print'\n CONFUSION MATRIX:'        
    print ConfusionMatrix(y_test, y_predicted)
    
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('reference', help='haspeede reference set')
    parser.add_argument('predicted', help='system output')
    args = parser.parse_args() 
    
    with open(sys.argv[1], 'r') as tf:
        y_test = preproc(tf)
        
    with open(sys.argv[2], 'r') as pf:
        y_predicted = preproc(pf)
        
    eval(y_test, y_predicted)
            
      
  