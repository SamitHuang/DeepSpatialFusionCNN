# e.g. python cross_evaluate.py --patches-overlap=1

import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix
import numpy as np
import itertools
import argparse

parser = argparse.ArgumentParser(description='Classification of breast cancer')
parser.add_argument('--patches-overlap', type=int, default=0, help= "overlap in image-wise/spatial network")
args= parser.parse_args()

start = 1
end = 10 

#TODO: this is not a good manner. relying on os command and not flexible 
CHK_PATH = "./checkpoints/resnet18_DNNvoter2018_fold"
NUM_VALIDATION = 40

cross_spatial = args.patches_overlap
if cross_spatial :
    RESULT_FILE = "./checkpoints/cross_res_resnet18_DNNvoter_2018_cross_spatial.csv"
    csv_file_name ="/test_result_cross_spatial.csv"
else:
    RESULT_FILE = "./checkpoints/cross_res_resnet18_DNNvoter_2018.csv"
    csv_file_name = "/test_result.csv"

def predict():
    for idx in range(start, end+1):
        if cross_spatial:
            os.system("python validate.py --patches-overlap=1 --fold-index="+str(idx))
        else:
            os.system("python validate.py --fold-index="+str(idx))

def merge():
    res_all=""
    # temporary implementation
    for idx in range(start, end+1):
        with open(CHK_PATH + str(idx) + csv_file_name) as fp :
            for line in fp:
                res_all = res_all + line 
    with open(RESULT_FILE,"w") as fp:
        fp.write(res_all)


def evulate(path):
    pred4=[]
    pred2=[]
    labels4=[]
    labels2=[]
    conf4=[]
    score2=[]
    with open(path) as fp:
        for line in fp:
            line = line[:-1]
            items = line.split("\t")
            fn = items[0]
            if("iv" in fn):
                labels4.append(3)
                labels2.append(1)
            elif("is" in fn):
                labels4.append(2)
                labels2.append(1)
            elif("b" in fn):
                labels4.append(1)
                labels2.append(0)
            elif("n" in fn):
                labels4.append(0)
                labels2.append(0)

            pred4.append(int(items[1]))
            conf4.append(float(items[2])/100.0)
            pred2.append(int(items[3]))
            label2_pred = int(items[3])
            conf_temp = float(items[4])/100.0
            prob_carcinoma= conf_temp if label2_pred ==1 else  1-conf_temp
            score2.append(prob_carcinoma)

    # average acc and std 
    acc4_list = np.zeros(10)
    acc2_list = np.zeros(10)
    std = 0
    for i in range(0,10):
        acc4_list[i] = accuracy_score(labels4[i*NUM_VALIDATION : (i+1)*NUM_VALIDATION], pred4[i*NUM_VALIDATION :(i+1)*NUM_VALIDATION ])  
        acc2_list[i] = accuracy_score(labels2[i*NUM_VALIDATION : (i+1)*NUM_VALIDATION], pred2[i*NUM_VALIDATION :(i+1)*NUM_VALIDATION])
    acc4_ave = acc4_list.mean()
    acc4_std = acc4_list.std()
    acc2_ave = acc2_list.mean()
    acc2_std = acc2_list.std()

    print("Average acc.(4-class) in 10-fold cross validation: {}".format(acc4_ave))
    print("STD: {}".format(acc4_std))
    print("Average acc.(2-class) in 10-fold cross validation: {}".format(acc2_ave))
    print("STD: {}".format(acc2_std))
   

    # overall accurarcy 
    acc4 = accuracy_score(labels4, pred4)
    cm = confusion_matrix(labels4, pred4)
    print("confusion_matrix:\r\n",cm, "\r\n accuaracy of 4-class ",acc4)
    acc2 = accuracy_score(labels2, pred2)
    cm2 = confusion_matrix(labels2, pred2)
    fpr, tpr, _ = roc_curve(labels2, score2)
    #print(fpr,tpr,_)
    AUC = auc(fpr, tpr)
    '''
    plt.figure()
    lw = 1.5
    plt.plot(fpr, tpr, color='darkorange',
	     lw=lw, label='AUC = %0.3f' % AUC)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    #plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
    print("confusion_matrix:\r\n",cm2, "\r\n accuaracy of 2-class ",acc2)
    print("AUC="+str(AUC))
    
    '''
    '''
    plt.figure()
    plot_confusion_matrix(cm,["Normal","Benign","InSitu","Invasive"], normalize = False, title = '')
    plt.show()
    '''

def plot_confusion_matrix(cm, classes,normalize=False, title='Confusion matrix',cmap=plt.cm.Blues):
        if(normalize):
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('Ground truth')
        plt.xlabel('Prediction')

if __name__ == "__main__":
    predict()
    merge()
    evulate(RESULT_FILE)


