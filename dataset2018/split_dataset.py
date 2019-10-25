#coding:utf-8
import random
import os
import math
import argparse
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--fold_idx', type=int, default=-1)
args = parser.parse_args()

#DATASET_PATH='/home/jason/MedicalAgent/BreastCancer/git_il/ICIAR2018/dataset2018'
DATASET_PATH = "/bigdata/BreastHistology/ICIAR2018/dataset2018"

#TRAIN_PATH='/home/jason/MedicalAgent/BreastCancer/git_il/ICIAR2018/dataset/train'

VALIDATION_PATH='./validation'

VALID_NUM=10 # 10 per class, total 40
labels=['Normal','Benign','InSitu','Invasive']

folds=[]
# e.g. 2..., responds to is002.tif, iv002.tif, nor002.tif, ben002.tif, 4 images per number. 400 images in total.
folds = [[2, 22, 40, 58, 71, 14, 32, 52, 67, 90], 
        [48, 56, 80, 11, 61, 12, 54, 51, 99, 9], 
        [83, 45, 77, 31, 95, 41, 15, 59, 38, 23], 
        [73, 21, 72, 36, 42, 1, 28, 50, 81, 8], 
        [49, 39, 3, 24, 68, 43, 29, 91, 18, 65], 
        [62, 96, 89, 70, 17, 44, 37, 97, 25, 10], 
        [53, 75, 30, 79, 66, 57, 55, 85, 82, 34], 
        [16, 98, 94, 60, 78, 69, 27, 33, 20, 7], 
        [92, 86, 47, 46, 84, 19, 93, 100, 63, 13], 
        [6, 87, 35, 88, 26, 76, 74, 4, 5, 64]]

def generate_fold_num():
    fold1=[2,22,40,58,71,14,32,52,67,90]
    left_idx = list(range(1,101)) 
    for i in fold1:
        left_idx.remove(i)

    for i in range(10):
        folds.append(fold1)

    #for f in range(9):
    FOLD_NUM = 10 
    cnt = 1
    in_cnt = 0
    for cnt in range(1,10):    
        folds[cnt] = random.sample(left_idx,10)
        for idx in folds[cnt]:
            left_idx.remove(idx)
        cnt += 1
    print(folds)

# from fold1 to fold10
def generate_fold_data(fold_idx):
    fdn = "fold"+str(fold_idx)
    cmd = "mkdir " + fdn + ";" \
        + "cd " + fdn + ";" \
        + "mkdir train validation;"  \
        + "cd train;" \
        + "mkdir Normal Benign InSitu Invasive;"\
        + "cd ../validation;"\
        + "mkdir Normal Benign InSitu Invasive all"
    os.system(cmd)
    
    validate_idx = folds[fold_idx-1]
    train_idx= list(range(1,101)) 
    for i in validate_idx:
        train_idx.remove(i)
    #print(validate_idx)
    #print(train_idx)
    for idx in validate_idx:
        idx_fmt= '{:03d}.tif '.format(idx)
        os.system('ln -s ' + DATASET_PATH + '/origin/Benign/b' + idx_fmt + DATASET_PATH + "/" + fdn + '/validation/Benign/.')
        os.system('ln -s ' + DATASET_PATH + '/origin/Normal/n' + idx_fmt + DATASET_PATH + "/" + fdn+ '/validation/Normal/.')
        os.system('ln -s ' + DATASET_PATH + '/origin/InSitu/is' + idx_fmt + DATASET_PATH + "/" + fdn+ '/validation/InSitu/.')
        os.system('ln -s ' + DATASET_PATH + '/origin/Invasive/iv' + idx_fmt + DATASET_PATH + "/" + fdn+ '/validation/Invasive/.')
    
    cmd2 = "cd " + fdn + "/validation;" \
        + "mkdir all;" \
        + "cp Benign/*.tif InSitu/*.tif Invasive/*.tif Normal/*.tif all/."
    os.system(cmd2)

    for idx in train_idx:
        idx_fmt= '{:03d}.tif '.format(idx)
        os.system('ln -s ' + DATASET_PATH + '/origin/Benign/b' + idx_fmt + DATASET_PATH + "/" + fdn + '/train/Benign/.')
        os.system('ln -s ' + DATASET_PATH + '/origin/Normal/n' + idx_fmt + DATASET_PATH + "/" + fdn+ '/train/Normal/.')
        os.system('ln -s ' + DATASET_PATH + '/origin/InSitu/is' + idx_fmt + DATASET_PATH + "/" + fdn+ '/train/InSitu/.')
        os.system('ln -s ' + DATASET_PATH + '/origin/Invasive/iv' + idx_fmt + DATASET_PATH + "/" + fdn+ '/train/Invasive/.')

def move_data_old():
    cnt = 0
    while(cnt <10):
#for i in range(2):
        idx = math.ceil(100*random.random())
        idx_fmt= '{:03d}.tif '.format(idx)
        #print(DATASET_PATH+'/Benign/b'+idx_fmt)
        #p1=DATASET_PATH+'/Benign/b'+idx_fmt
        #print(os.path.exists(p1[:-1])) #(DATASET_PATH + '/Benign/b021.tif'))
        if(os.path.exists((DATASET_PATH+'/Benign/b'+idx_fmt)[:-1])):
            print('moving ' + idx_fmt)
            os.system('mv ' + DATASET_PATH + '/Benign/b' + idx_fmt + VALIDATION_PATH + '/Benign/.')
            os.system('mv ' + DATASET_PATH + '/Normal/n' + idx_fmt + VALIDATION_PATH + '/Normal/.')
            os.system('mv ' + DATASET_PATH + '/InSitu/is' + idx_fmt + VALIDATION_PATH + '/InSitu/.')
            os.system('mv ' + DATASET_PATH + '/Invasive/iv' + idx_fmt + VALIDATION_PATH + '/Invasive/.')
            cnt = cnt + 1


#os.system('ln -s ' + DATASET_PATH + " " + TRAIN_PATH)
if __name__ == "__main__":
    #generate_fold_num()
    generate_fold_data(args.fold_idx)

