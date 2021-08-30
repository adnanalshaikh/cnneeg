
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, \
                            multilabel_confusion_matrix, confusion_matrix, roc_auc_score, roc_curve, auc

import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle

class Measures(object):
    
    def __init__(self, model, x_test, y_test):

        self.y_true  = y_test
        self.x_test  = x_test
        self.y_pred  = model.predict(x_test)
        self.model   = model 
        self.measures = dict()
               
    def eval(self):
         
        self.measures['loss']  = self.model.evaluate(self.x_test, self.y_true)[0]
        
        y_true = np.argmax(self.y_true, axis = 1)
        y_pred = np.argmax(self.y_pred, axis = 1)
              
        self.measures['accuracy']            = accuracy_score( y_true, y_pred)
        self.measures['precision_macro']      = precision_score( y_true, y_pred, average='macro' )
        self.measures['recall_macro']         = recall_score(y_true, y_pred, average='macro' )
        self.measures['f1Score_macro']       = f1_score(y_true, y_pred, average='macro')
        
        self.measures['precision_micro']      = precision_score( y_true, y_pred, average='micro' )
        self.measures['recall_micro']         = recall_score(y_true, y_pred, average='micro' )
        self.measures['f1Score_micro']        = f1_score(y_true, y_pred, average='micro')
                
        self.measures['aucs_ovo_macro']       = roc_auc_score(self.y_true, self.y_pred, multi_class="ovo", average='macro')
        self.measures['aucs_ovo_weighted']    = roc_auc_score(self.y_true, self.y_pred, multi_class="ovo", average='weighted')
        self.measures['aucs_ovr_macro']       = roc_auc_score(self.y_true, self.y_pred, multi_class="ovr", average='macro')
        self.measures['aucs_ovr_weighted']    = roc_auc_score(self.y_true, self.y_pred, multi_class="ovr", average='weighted')
        
        self.measures['mcm']                  = multilabel_confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
        self.measures['cm']                   = confusion_matrix(y_true, y_pred)
                        
        return self.measures

    def ROCCurve(self):
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        n_classes = 3
        
        for i in range(n_classes):
            fpr[i], tpr[i], thresholds = roc_curve(self.y_true[:, i], self.y_pred[:, i], drop_intermediate=False)
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(self.y_true.ravel(), self.y_pred.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        
        # Compute macro-average ROC curve and ROC area
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

        # Interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        
        for i in range(n_classes):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])

        mean_tpr /= n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
        self.plotRocCurves(fpr, tpr, roc_auc, n_classes)

    def plotRocCurves(self, fpr, tpr, roc_auc, n_classes):

        fig1 = plt.figure(1)
        lw=2

        colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
        
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=lw, label='class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC curve')
        plt.legend(loc="lower right")
        plt.show()
        plt.close(fig1)
