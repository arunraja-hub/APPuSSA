# Automatic Punctuation Prediction using Sequence to Sequence Approaches (APPuSSA)

* Tables of experiments implemented are as follows:
![image](https://user-images.githubusercontent.com/43485111/116096065-89825200-a6db-11eb-8741-c44dfcf7889e.png)
* The scripts can be independently run as they were developed on Google Colab first due to lack of access to a reliable GPU
* Due to lack of F1 score metric in Keras (https://github.com/keras-team/keras/wiki/Keras-2.0-release-notes#losses--metrics), I have created a F1 metric using Keras' Metric API for both F1-micro and F1-class for future use in punctuation prediction. They can be easily edited for used in other tasks as well.
* The MGB dataset (https://ieeexplore.ieee.org/document/7404863) was used for the experiments. The required files from this dataset are train.txt, dev.txt, train.ark, dev.ark, train.ctm and dev.ctm
