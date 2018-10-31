# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 14:42:43 2018

@author: arvis
"""

from tool import use_classifier_to_kfold as clf_kfold
import argparse

def usage():
    print('example : python clf_to_kfold.py CV -f Instagram_feature_word2vec.csv -y Instagram_labeled.csv -c SVM -t 10 -s 10cv_results')
    #return
def savefile(data,name):
    
    data.to_csv(name+'.csv',encoding="UTF-8")
    print("save finish")
        
        
def get_CV(X, y ,classifier, cv_times):

    run_cv = clf_kfold(X, y ,classifier, cv_times)
    return run_cv._cross_validation()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='words preparation',
                                     usage=usage())
    
    parser.add_argument('CV',
                        help='options: [CV]')
    
    
    parser.add_argument('-f', '--features',
                        type=str,
                        help='read X features data name example:Instagram_feature_word2vec.csv')

    parser.add_argument('-y', '--labeled',
                        type=str,
                        help='read y labeled data files name example:Instagram_labeled.csv')

    
    parser.add_argument('-c', '--classifier',
                        type=str,
                        help='input classifier you want (you can use:SVM,KNN,LDA,DecisionTree,MultinomialNB,GaussianNB)')
    
    parser.add_argument('-t', '--times', type=int, help='this set is random_state times (ex: set(10) will run random_state(0-9) 10CV  get 100 different result )  ')
    
    parser.add_argument('-s', '--save', help='save file name')
    
    args = parser.parse_args()
    
    if args.CV:
        savefile(get_CV(args.features, args.labeled, args.classifier, args.times), args.save)
        
    else:
        usage()