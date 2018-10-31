# ML_Classifier_kfolds

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
Start Cross_Validation...
CV_times : 0
0.9640718562874252
0.976
0.976
0.97
0.958
0.952
0.95
0.96
0.964
0.9719438877755511
CV_times : 1
0.9680638722554891
0.962
0.962
0.946
0.964
0.958
0.964
0.978
0.972
0.9619238476953907
CV_times : 2
0.9700598802395209
0.962
0.96
0.962
0.962
0.966
0.96
0.962
0.966
0.9679358717434869
CV_times : 3
0.9700598802395209
0.954
0.972
0.956
0.97
0.972
0.964
0.968
0.96
0.9539078156312625
CV_times : 4
0.9720558882235529
0.976
0.964
0.974
0.956
0.966
0.954
0.968
0.948
0.9599198396793587
CV_times : 5
0.9500998003992016
0.96
0.972
0.972
0.97
0.96
0.97
0.956
0.964
0.9659318637274549
CV_times : 6
0.9540918163672655
0.968
0.972
0.974
0.958
0.95
0.972
0.972
0.97
0.9579158316633266
CV_times : 7
0.9680638722554891
0.96
0.964
0.956
0.964
0.962
0.958
0.97
0.96
0.9679358717434869
CV_times : 8
0.9680638722554891
0.956
0.958
0.968
0.956
0.978
0.954
0.964
0.962
0.9719438877755511
CV_times : 9
0.9660678642714571
0.96
0.956
0.958
0.966
0.974
0.964
0.964
0.958
0.9659318637274549
save finish

```
