# ML_Classifier_kfolds
# ML_CrossValidation

本程式是用來跑10次交叉驗證(Cross_Validation)分數並將結果F1分數存入CSV

可使用的分類器(Classifier)有:SVM、LDA、KNN、DecisionTree、MultinomialNB與GaussianNB

使用環境為python3 ， 使用前須先自行安裝sklearn套件


# 程式使用方法  
```
python clf_to_kfold.py CV -f Instagram_feature_word2vec.csv -y Instagram_labeled.csv -c GaussianNB -t 10 -s 10cv_results

-f = features data(檔案名稱) *(csv)  

-y = labeled data(檔案名稱) *(csv)  

-c = classifier(分類器選擇) you can use: SVM, KNN, LDA, DecisionTree, MultinomialNB, GaussianNB  

-t = times(跑t次的10次交叉驗證) 如輸入10次 則會下0-9的random_state進行10次的10個Cross_Validation 則會得到100個F1-Score  

-s = save 處理完成後儲存的檔案名稱 (會存成CSV檔案，如GIT上.CSV所示)  

如果使用的classifier是SVM的話，可供四種kernel選擇，1.rbf 2.linear 3.poly 4.sigmoid  

其中RBF kernel可以選擇需要的參數(parameter) C and gamma  

happy to wait result  
```

# example use:  
```
python clf_to_kfold.py CV -f Instagram_feature_word2vec.csv -y Instagram_labeled.csv -c GaussianNB -t 10 -s 10cv_results

your datasets X(fearures) X.head() :
   A3_token  A4_emoji  A5_tag  A6_num  A7_sign    ...     text_295  text_296  text_297  text_298  text_
0        15         3       6       0        0    ...    -0.585410  0.115853  0.212104  0.692878  0.169
1        14         0      22       0        0    ...    -0.356731  0.136205  0.137098  0.531787  0.143
2        14         0      22       0        0    ...    -0.358607  0.115694  0.097615  0.497731  0.194
3         2         0      25       0        0    ...     0.000000  0.000000  0.000000  0.000000  0.000
4         2         0      25       0        0    ...     0.000000  0.000000  0.000000  0.000000  0.000

[5 rows x 308 columns]
your datasets y(labeled) y.head() :
     y
0  0.0
1  1.0
2  1.0
3  1.0
4  1.0
Start Cross_Validation...
CV_times : 0
0.7265469061876247
0.664
0.682
0.696
0.686
0.634
0.732
0.718
0.698
0.7034068136272545
CV_times : 1
0.6926147704590818
0.736
0.71
0.696
0.73
0.656
0.7
0.676
0.668
0.685370741482966
CV_times : 2
0.7784431137724551
0.674
0.734
0.678
0.698
0.674
0.694
0.696
0.67
0.6893787575150301
CV_times : 3
0.6826347305389222
0.668
0.704
0.68
0.692
0.694
0.71
0.69
0.738
0.655310621242485
CV_times : 4
0.6986027944111777
0.656
0.684
0.734
0.744
0.674
0.684
0.684
0.676
0.6793587174348698
CV_times : 5
0.7065868263473054
0.666
0.698
0.68
0.688
0.686
0.7
0.73
0.656
0.7575150300601202
CV_times : 6
0.7125748502994012
0.666
0.674
0.674
0.742
0.686
0.696
0.7
0.728
0.657314629258517
CV_times : 7
0.7245508982035929
0.73
0.696
0.69
0.684
0.686
0.686
0.676
0.688
0.6813627254509018
CV_times : 8
0.7025948103792415
0.732
0.748
0.684
0.664
0.668
0.686
0.638
0.684
0.7254509018036072
CV_times : 9
0.7045908183632734
0.642
0.694
0.678
0.692
0.724
0.694
0.7
0.716
0.6833667334669339
save finish

```
