import time
import ntpath
import datetime
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as ply
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix
from sklearn.preprocessing import label_binarize
from tensorboardX import SummaryWriter
from .datasets import *
import torch.nn as nn
from torchnet import meter
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

TRAIN_PATH = '/train'
VALIDATION_PATH = '/validation'

feature_map_size = 10 
NUM_CLASSES=4

class BaseModel:
    def __init__(self, args, network, weights_path):
        self.args = args
        self.weights = weights_path
        self.network = network.cuda() if args.cuda else network
        self.load()

    def load(self):
        try:
            if os.path.exists(self.weights):
                print('==> Loading model from ' + self.weights )
                self.network.load_state_dict(torch.load(self.weights))
        except:
            print('Failed to load pre-trained network')

    def save(self, epoch=""):
        print('Saving model to {}'.format(self.weights))
        torch.save(self.network.state_dict(), self.weights)

net_type = "cnn"

class PatchWiseModel(BaseModel):
    def __init__(self, args, network):
        super(PatchWiseModel, self).__init__(args, network, args.checkpoints_path + '/weights_' + network.name() + '.pth')

    def train(self):
        self.network.train()
        print('Start training patch-wise network: {}\n'.format(time.strftime('%Y/%m/%d %H:%M')))

        #self.writer = SummaryWriter(self.args.checkpoints_path + '/log')
        #self.writer.add_graph(self.network,Variable(torch.rand(self.args.batch_size, 3, 2048, 1536)) )
        train_loader = DataLoader(
            dataset=PatchWiseDataset(path=self.args.dataset_path + TRAIN_PATH, stride=self.args.patch_stride, rotate=True, flip=True, enhance=True),
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=4
        )
        optimizer = optim.Adam(self.network.parameters(), lr=self.args.lr, betas=(self.args.beta1, self.args.beta2))
        #optimizer = optim.Adam(self.network.classifier.parameters(), lr=self.args.lr, betas=(self.args.beta1, self.args.beta2))
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
        loss_temp, best = self.validate(verbose=False)
        mean = 0
        epoch = 0
        best_epoch = -1
        loss_in_best_epoch = 0
        for epoch in range(1, self.args.epochs + 1):

            self.network.train()
            scheduler.step()
            stime = datetime.datetime.now()

            correct = 0
            total = 0

            for index, (images, labels) in enumerate(train_loader):
                if self.args.cuda:
                    images, labels = images.cuda(), labels.cuda()

                optimizer.zero_grad()
                output = self.network(Variable(images))
                loss = F.nll_loss(output, Variable(labels))
                loss.backward()
                optimizer.step()

                _, predicted = torch.max(output.data, 1)
                correct += torch.sum(predicted == labels)
                total += len(images)

                if index > 0 and index % self.args.log_interval == 0:
                    print('Epoch: {}/{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accuracy: {:.2f}%'.format(
                        epoch,
                        self.args.epochs,
                        index * len(images),
                        len(train_loader.dataset),
                        100. * index / len(train_loader),
                        loss.item(),
                        100 * correct / total
                    ))

            print('\nEnd of epoch {}, time: {}'.format(epoch, datetime.datetime.now() - stime))
            test_loss, acc = self.validate()
            mean += acc
            if acc >= best:
                best = acc
                best_epoch = epoch
                loss_in_best_epoch = test_loss
            self.save(epoch)

            self.writer.add_scalars('loss', {"train":loss.data[0], "validate":test_loss},epoch)
            self.writer.add_scalars('acc', {"train" : 100 * correct / total, "validate":acc },epoch)
        print('\nEnd of training, best accuracy: {} at epoch {}, loss in this epoch: {}\n'.format(best, best_epoch, loss_in_best_epoch))
        self.writer.close()
        return [loss_in_best_epoch, best, best_epoch]

    def validate(self, verbose=True):
        self.network.eval()

        test_loss = 0
        correct = 0
        classes = len(LABELS)

        test_loader = DataLoader(
            dataset=PatchWiseDataset(path=self.args.dataset_path + VALIDATION_PATH, stride=self.args.patch_stride),
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=4
        )
        num_test = len(test_loader.dataset)

        if verbose:
            print('\nEvaluating....')
        
        labels_all = np.empty(0)
        preds_all = np.empty(0)
        for images, labels in test_loader:
            if self.args.cuda:
                images, labels = images.cuda(), labels.cuda()
            
            output = self.network(Variable(images, volatile=True))
            test_loss += F.nll_loss(output, Variable(labels), size_average=False).item()
            _, predicted = torch.max(output.data, 1)
            correct += torch.sum(predicted == labels)
            labels_all = np.append(labels_all, labels.cpu().numpy())
            preds_all = np.append(preds_all, predicted.cpu().numpy())


        test_loss /= len(test_loader.dataset)
        #acc = 100. * correct / len(test_loader.dataset)
        acc = accuracy_score(labels_all, preds_all) 
        precision, recall, f1, _ = precision_recall_fscore_support(labels_all, preds_all, average=None, labels=[0,1,2,3])

        if verbose:
            print('Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
                test_loss,
                correct,
                len(test_loader.dataset),
                100*acc 
            ))
            for label in range(classes):
                print('{}:  \t Precision: {:.2f},  Recall: {:.2f},  F1: {:.2f}'.format(
                    LABELS[label],
                    precision[label],
                    recall[label],
                    f1[label]
                ))
            print('')
        return test_loss, acc

    def output(self, input_tensor):
        self.network.eval()
        res = self.network.features(Variable(input_tensor, volatile=True))
        return res.squeeze()
    
    def output_prob(self, input_tensor):
        self.network.eval()
        res = self.network(Variable(input_tensor, volatile=True))
        return res.squeeze()
    
    def visualize(self, path, channel=0):
        self.network.eval()
        dataset = TestDataset(path=path, stride=PATCH_SIZE)
        data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)

        for index, (image, file_name) in enumerate(data_loader):

            if self.args.cuda:
                image = image[0].cuda()

            patches = self.output(image)

            output = patches.cpu().data.numpy()

            map = np.zeros((3 * 64, 4 * 64))

            for i in range(12):
                row = i // 4
                col = i % 4
                map[row * 64:(row + 1) * 64, col * 64:(col + 1) * 64] = output[i]

            if len(map.shape) > 2:
                map = map[channel]

            with Image.open(file_name[0]) as img:
                ply.subplot(121)
                ply.axis('off')
                ply.imshow(np.array(img))

                ply.subplot(122)
                ply.imshow(map, cmap='gray')
                ply.axis('off')

                ply.show()


class ImageWiseModel(BaseModel):
    def __init__(self, args, image_wise_network, patch_wise_network):
        if(args.patches_overlap):
            model_weights_file = args.checkpoints_path + '/weights_' + image_wise_network.name() + '.pth_cross_spatial'
        else:
            model_weights_file = args.checkpoints_path + '/weights_' + image_wise_network.name() + '.pth'
        super(ImageWiseModel, self).__init__(args, image_wise_network, model_weights_file)

        self.patch_wise_model = PatchWiseModel(args, patch_wise_network)
        self._test_loader = None
        
        #args.patches_overlap=True

    def train(self):

        self.network.train()
        print('Evaluating patch-wise model...')
        print(self.args.dataset_path + TRAIN_PATH)
        train_loader = self._patch_loader(self.args.dataset_path + TRAIN_PATH, True)

        print('Start training image-wise network: {}\n'.format(time.strftime('%Y/%m/%d %H:%M')))

        optimizer = optim.Adam(self.network.parameters(), lr=self.args.lr, betas=(self.args.beta1, self.args.beta2))
        best = self.validate(verbose=False)
        mean = 0
        epoch = 0
        best_epoch = -1
        for epoch in range(1, self.args.epochs + 1):

            self.network.train()
            stime = datetime.datetime.now()

            correct = 0
            total = 0

            for index, (images, labels) in enumerate(train_loader):

                if self.args.cuda:
                    images, labels = images.cuda(), labels.cuda()

                optimizer.zero_grad()
                output = self.network(Variable(images))
                loss = F.nll_loss(output, Variable(labels))
                loss.backward()
                optimizer.step()

                _, predicted = torch.max(output.data, 1)
                correct += torch.sum(predicted == labels)
                total += len(images)

                if index > 0 and index % 10 == 0:
                    print('Epoch: {}/{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accuracy: {:.2f}%'.format(
                        epoch,
                        self.args.epochs,
                        index * len(images),
                        len(train_loader.dataset),
                        100. * index / len(train_loader),
                        loss.data[0],
                        100 * correct / total
                    ))

            print('\nEnd of epoch {}, time: {}'.format(epoch, datetime.datetime.now() - stime))
            acc = self.validate()
            mean += acc
            if acc >= best:
                best = acc
                best_epoch = epoch
            self.save(epoch)

        print('\nEnd of training, best accuracy: {} at epoch {}, mean accuracy: {}\n'.format(best, best_epoch, mean / epoch))

        return [best, best_epoch]

    def validate_voter(self, verbose=True):
        correct = 0
        classes = len(LABELS)

        tp = [0] * classes
        fp = [0] * classes
        tpfp = [0] * classes
        tpfn = [0] * classes
        precision = [0] * classes
        recall = [0] * classes
        f1 = [0] * classes
        voter= "maj"

        data_loader = self._patch_loader(self.args.dataset_path + VALIDATION_PATH, False)
        stime = datetime.datetime.now()

        if verbose:
            print('\t sum\t\t max\t\t maj')

        res = []

        for probs_map, labels in data_loader:
            if self.args.cuda:
                probs_map, labels = probs_map.cuda(), labels.cuda()
            #probs_maps, shape [32,12,4]  are maps of prob of four classes of 12 patches each in 32 images
            print(type(probs_map))
            for i, prob_patches in enumerate(probs_map):
                label = labels[i]
                #print(prob_patches.cpu().numpy())
                if(voter=="maj"):
                    _, predicted = torch.max(prob_patches, 1)
                    maj_prob = 3 - np.argmax(np.sum(np.eye(4)[np.array(predicted).reshape(-1)], axis=0)[::-1])
                    class_pred = maj_prob
                elif (voter=="prior"):
                    _, predicted = torch.max(prob_patches, 1)
                    #0 Normal 1.. 2. 3 Invasive, the larger index, the severer or higher prior
                    class_pred = np.array(predicted).max()

                tpfp[class_pred] += 1
                tpfn[label] += 1
                if(class_pred == label):
                    tp[label] += 1
                    correct += 1
        
        print('Accuracy: {}/{} ({:.2f}%)'.format(
                correct,
                len(data_loader.dataset),
                100. * correct / len(data_loader.dataset)
            ))

        for label in range(classes):
            precision[label] = (tp[label]/(tpfp[label]+1e-8))
            recall[label] = (tp[label]/(tpfn[label]+1e-8))
            f1[label] = 2 * precision[label] * recall[label] / (precision[label] + recall[label] + 1e-8)
            print('{}:  \t Precision: {:.2f},  Recall: {:.2f},  F1: {:.2f}'.format(
                    LABELS[label],
                    precision[label],
                    recall[label],
                    f1[label]
                ))
                

    def validate(self, verbose=True, roc=False):
        self.network.eval()

        if self._test_loader is None:
            print("DEBUG: " + self.args.dataset_path + VALIDATION_PATH)
            self._test_loader = self._patch_loader(self.args.dataset_path + VALIDATION_PATH, False)

        val_loss = 0
        correct = 0
        classes = len(LABELS)

        tp = [0] * classes
        tpfp = [0] * classes
        tpfn = [0] * classes
        precision = [0] * classes
        recall = [0] * classes
        f1 = [0] * classes
        
        
        if verbose:
            print('\nEvaluating....')

        labels_true = []
        labels_pred = np.empty([0, 4])
        predicteds = np.empty([0,0]) 
        scores = np.empty([0,4]) 
        for images, labels in self._test_loader:

            if self.args.cuda:
                images, labels = images.cuda(), labels.cuda()

            output = self.network(Variable(images, volatile=True))

            val_loss += F.nll_loss(output, Variable(labels), size_average=False).item()
            _, predicted = torch.max(output.data, 1)
            correct += torch.sum(predicted == labels)
            predicteds = np.append(predicteds, predicted.cpu().numpy())
            labels_true = np.append(labels_true, labels.cpu().numpy())
            labels_pred = np.append(labels_pred, torch.exp(output.data).cpu().numpy(), axis=0)
       
        val_loss /= len(self._test_loader.dataset)
        acc = 100 * accuracy_score(labels_true, predicteds)
        cm= confusion_matrix(labels_true, predicteds)
        print("==> Confusion Matrix: \r\n",cm)
        precision, recall, f1, _ = precision_recall_fscore_support(labels_true, predicteds, average=None, labels=[0,1,2,3])
        #precision_avg, recall_avg, f1_avg, _ = 
        print("precsion, recall, f1:", precision_recall_fscore_support(labels_true, predicteds, average="macro"))

        #plt.figure()
        #plot_confusion_matrix(cm, classes=4,title='Confusion matrix, without normalization')
        #plt.show()
        if verbose:
            print('Average loss: {:.4f}, 4-class Accuracy: {}/{} ({:.2f}%)'.format(
                val_loss,
                correct,
                len(self._test_loader.dataset),
                acc
            ))

            for label in range(classes):
                print('{}:  \t Precision: {:.2f},  Recall: {:.2f},  F1: {:.2f}'.format(
                    LABELS[label],
                    precision[label],
                    recall[label],
                    f1[label]
                ))
       
    # calc 2 classes metrics 
        score_carcinoma = np.empty(labels_pred.shape[0])
        score_carcinoma = labels_pred[:,2]+labels_pred[:,3]
        labels_carcinoma = np.array((labels_true>1),dtype=int) # 2,3 carinoma
        fpr, tpr, _ = roc_curve(labels_carcinoma, score_carcinoma)
        #print(fpr,tpr,_)
        AUC = auc(fpr, tpr)
        prediction2 = np.array((score_carcinoma > 0.5),dtype=int)  #set THRESHOLD 0. 
        predicted2 = np.array((predicteds>1), dtype=int)
        #print(predicteds, "\r\n",predicted2, "\r\n", labels_true)
        ACC2 = accuracy_score(labels_carcinoma, predicted2) 
        #print(prediction2)
        #print(labels_carcinoma)
    
        print("2-classes classification AUC: {:.2%},  Accurary: {:.2%}".format(AUC,ACC2))
        #log the result to file
        

        labels_true = label_binarize(labels_true, classes=range(classes))
        for lbl in range(classes):
            fpr, tpr, _ = roc_curve(labels_true[:, lbl], labels_pred[:, lbl])
            #roc_auc[lbl] = auc(fpr, tpr)
        #print("AUC: {:.3f}".format(roc_auc))
        
        if roc == 1:
            labels_true = label_binarize(labels_true, classes=range(classes))
            for lbl in range(classes):
                fpr, tpr, _ = roc_curve(labels_true[:, lbl], labels_pred[:, lbl])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, lw=2, label='{} (AUC: {:.1f})'.format(LABELS[lbl], roc_auc * 100))
            
            plt.xlim([0, 1])
            plt.ylim([0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.legend(loc="lower right")
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.title('Receiver Operating Characteristic')
            plt.show()

        
        return acc
    # evulate and save result:
    def evaluate(self, path, verbose=True, ensemble=True):
        self.network.eval()
        if(self.args.patches_overlap==True):
            stride_ = PATCH_SIZE/2
        else:
            stride_ = PATCH_SIZE

        dataset = TestDataset(path=path, stride=stride_, augment=ensemble)
        data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
        stime = datetime.datetime.now()
        res = []
        scores = np.empty([0,4])
        labels = []
        predictions=[]
        for index, (image, file_name) in enumerate(data_loader):
            n_bins, n_patches = image.shape[1], image.shape[2] # n_bins=8 from rotation and flipping, n_patches dependos on stride
            image = image.view(-1, 3, PATCH_SIZE, PATCH_SIZE)

            if self.args.cuda:
                image = image.cuda()

            #res = res.view((bsize, -1, 4)).data.cpu().numpy()
            patches = self.patch_wise_model.output_prob(image)
            patches = patches.view(n_bins, -1, 4)
            patches = patches.cuda()
            output = self.network(patches)
            _, predicted = torch.max(output.data, 1)
            #TODO: Here use majority ensemble for 8 images output predictions. What about avereage ensemble (average then normalize to 1) or other methods?  
            maj_prob = 3 - np.argmax(np.sum(np.eye(4)[np.array(predicted).reshape(-1)], axis=0)[::-1])

            confidence = np.sum(np.array(predicted) == maj_prob) / n_bins if ensemble else torch.max(torch.exp(output.data))
            confidence = np.round(confidence * 100, 2)
            # 2 class confidence
            predicted2 = np.array((predicted >1),dtype=int) # 2,3 carinoma
            if(ensemble):
                #TODO: 2-class majority vote need to be implement independently if ensemble
                confidence2 = np.sum(np.array(predicted2) == maj_prob2) / n_bins 
            else:
                maj_prob2= 1 if maj_prob > 1 else 0
                prob4 = torch.exp(output.data) # we use log_softmax in the network
                prob4 = np.array(prob4).reshape(4)
                if(maj_prob2 == 1):
                    confidence2 = prob4[2] + prob4[3]
                else:
                    confidence2 = prob4[0] + prob4[1]
            confidence2 = np.round(confidence2 * 100, 2)
            #predicted2 = np.array((predicted2))
            # ensemble all 8 augmented image prediction by average socre
            score = torch.exp(output.data).cpu().numpy()
            score = np.sum(score,axis=0) # sum up non-norm prob by row
            score = score / np.sum(score) # normalize by ai=ai/Sum(A)
            scores=np.append(scores, score)
            
            file_name = file_name[0].split("/")[-1]
            if("iv" in file_name):
                labels.append(3)
            elif("is" in file_name):
                labels.append(2)
            elif("b" in file_name):
                labels.append(1)
            elif("n" in file_name):
                labels.append(0)
            predictions.append(maj_prob)
            print(file_name, maj_prob, confidence, maj_prob2, confidence2) #, np.array(output.data)
            res.append([file_name, maj_prob, confidence, maj_prob2, confidence2])

        # 4-classes
        #print(res)
        acc4 = accuracy_score(labels, predictions)
        cm = confusion_matrix(labels, predictions)
        print("confusion_matrix:\r\n",cm, "\r\n accuaracy of 4-class ",acc4)
 
        # 2-classes
        #print(scores)
        #ACC2 = accuracy_score(labels_carcinoma, predicted2) 
        return res

    def _patch_loader(self, path, augment):
        images_path = '{}/{}_images.npy'.format(self.args.checkpoints_path, self.network.name())
        labels_path = '{}/{}_labels.npy'.format(self.args.checkpoints_path, self.network.name())

        if self.args.debug and augment and os.path.exists(images_path):
            np_images = np.load(images_path)
            np_labels = np.load(labels_path)

        else:
            if(self.args.patches_overlap==True):
                stride_ = PATCH_SIZE/2
            else:
                stride_ = PATCH_SIZE
            dataset = ImageWiseDataset(
                path=path,
                stride=stride_,
                flip=augment,
                rotate=augment,
                enhance=augment)
            
            # batch size, need to set to be 2 if use cross spatial/patches overlapping in image-wise netowrk. otherwise, we can set it to 4. 
            bsize = 2 #4 
            output_loader = DataLoader(dataset=dataset, batch_size=bsize, shuffle=True, num_workers=4)
            output_images = []
            output_labels = []

            for index, (images, labels) in enumerate(output_loader):
                if index > 0 and index % 10 == 0:
                    print('{} images loaded'.format(int((index * bsize) / dataset.augment_size)))

                if self.args.cuda:
                    images = images.cuda()
                #8 images, 8*12=96 patches, input: 96, 3, 512 ,512. --patchwise cnn feature map-->  96,16,16, ---view()---> 8,12,16,16
                #res = self.patch_wise_model.output(images.view((-1, 3, 512, 512)))
                #res = res.view((bsize, -1, feature_map_size,feature_map_size )).data.cpu().numpy()
                res = self.patch_wise_model.output_prob(images.view((-1, 3, 512, 512)))
                res = res.view((bsize, -1, 4)).data.cpu().numpy()
                #print("shape is " + res.shape)
                for i in range(bsize):
                    output_images.append(res[i])
                    output_labels.append(labels.numpy()[i])

            np_images = np.array(output_images)
            np_labels = np.array(output_labels)

            if self.args.debug and augment:
                np.save(images_path, np_images)
                np.save(labels_path, np_labels)
        images, labels = torch.from_numpy(np_images), torch.from_numpy(np_labels).squeeze()

        return DataLoader(
            dataset=TensorDataset(images, labels),
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=2
        )

    
