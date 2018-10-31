# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 14:42:43 2018

@author: arvis
"""

import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score

class use_classifier_to_kfold(object):
    def __init__(self, feature_data_X, labeled_data_y, classifier, cvtimes):
        self.feature_data_X = feature_data_X
        self.labeled_data_y = labeled_data_y
        self.classifier = classifier
        self.cvtimes = cvtimes 
        self.sum_predict = list()

    #選擇需要的分類器，以及需要跑幾輪的十次CV 
    def _cross_validation(self):
        
        X = pd.read_csv(self.feature_data_X, encoding='utf-8', index_col= False)
        print('your datasets X(fearures) X.head() :')
        print(X.head())
        X = X.loc[:,:].values
        
        y = pd.read_csv(self.labeled_data_y, encoding='utf-8', index_col= False)
        print('your datasets y(labeled) y.head() :')
        print(y.head())
        y = y.loc[:,:].values
        y = y.reshape(-1)

        print('Start Cross_Validation...')


        #classifier選項
        if self.classifier == 'SVM':
            kernel_dict = {'1':'rbf', '2':'linear', '3':'poly', '4':'sigmoid'}
            _kernel = str(input("input (number:1, 2, 3, 4) your kernel 1.rbf 2.linear 3.poly 4.sigmoid : "))
            if _kernel == '1':
                _C = input("setting C parameter you need (ex:10) :")
                _gamma = input("setting gamma parameter you need ex:(1) :")
                clf = SVC(kernel=kernel_dict[_kernel], C=float(_C), gamma=float(_gamma))
                self.run_Stratified_kfold(X, y, clf)
            else:
                clf = SVC(kernel=kernel_dict[_kernel])
                self.run_Stratified_kfold(X, y, clf)

        elif self.classifier == 'LDA':
            clf = LDA()
            self.run_Stratified_kfold(X, y, clf)
        elif self.classifier == 'KNN':
            clf = KNeighborsClassifier()
            self.run_Stratified_kfold(X, y, clf)

        elif self.classifier == 'DecisionTree':
            clf = DecisionTreeClassifier()
            self.run_Stratified_kfold(X, y, clf)

        elif self.classifier == 'MultinomialNB':
            clf = MultinomialNB()
            self.run_Stratified_kfold(X, y, clf)

        elif self.classifier == 'GaussianNB':
            clf = GaussianNB()
            self.run_Stratified_kfold(X, y, clf)

        data_dict = {self.classifier : self.sum_predict}
        df = pd.DataFrame(data_dict)
        return df

    def run_Stratified_kfold(self, X, y, clf):

        for cv_times in range(self.cvtimes):
            print('CV_times : {}'.format(cv_times))
            kf = StratifiedKFold(y, n_folds = self.cvtimes, random_state = cv_times, shuffle = True)
            for train_index, test_index in kf:
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                clf.fit(X_train, y_train)
                y_predict = clf.predict(X_test)
                print(sum(y_predict == y_test) / float(len(y_test)))
                self.sum_predict.append(f1_score(y_test, y_predict, average='macro'))

 