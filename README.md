# Automatic Punctuation Prediction using Sequence to Sequence Approaches (APPuSSA)

* Tables of experiments implemented are as follows:
![image](https://user-images.githubusercontent.com/43485111/116108234-4d082380-a6e6-11eb-8fd2-3bf0f733dfb6.png)
* The scripts can be run independently as they were developed on Google Colab first due to lack of access to a reliable GPU
* Due to lack of F1 score metric in Keras (https://github.com/keras-team/keras/wiki/Keras-2.0-release-notes#losses--metrics), I have created a F1 metric using Keras' Metric API for both F1-micro and F1-class for use in future work in punctuation prediction. They can be easily edited for use in other tasks as well.
* The MGB dataset (https://ieeexplore.ieee.org/document/7404863) was used for the experiments. The required files from this dataset are train.txt, dev.txt, train.ark, dev.ark, train.ctm and dev.ctm
