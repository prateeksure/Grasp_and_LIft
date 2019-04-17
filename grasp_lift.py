import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


#############function to read data###########
FNAME = "/home/prateek/Downloads/input/{0}/subj{1}_series{2}_{3}.csv"
def load_data(subj, series=range(1,9), prefix = 'train'):
    data = [pd.read_csv(FNAME.format(prefix,subject,s,'data'), index_col=0) for s in series]
    idx = [d.index for d in data]
    data = [d.values.astype(float) for d in data]
    if prefix == 'train':
        events = [pd.read_csv(FNAME.format(prefix,subject,s,'events'), index_col=0).values for s in series]
        return data, events
    else:
        return data, idx

def compute_features(X, scale=None):
    X0 = [x[:,0] for x in X]
    X = np.concatenate(X, axis=0)
    '''
    >>> a = np.array([[1, 2], [3, 4]])
    >>> b = np.array([[5, 6]])
    >>> np.concatenate((a, b), axis=0)
    array([[1, 2],
       [3, 4],
       [5, 6]])
    '''
    F = [];
    for fc in np.linspace(0,1,11)[1:]: 
        """
        >>> np.linspace(2.0, 3.0, num=5)
        array([ 2.  ,  2.25,  2.5 ,  2.75,  3.  ])
        """
        b,a = butter(3,fc/250.0,btype='lowpass')  
        '''
        https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.signal.butter.html
        '''
        F.append(np.concatenate([lfilter(b,a,x0) for x0 in X0], axis=0)[:,np.newaxis])
        """
        >>> np.append([1, 2, 3], [[4, 5, 6], [7, 8, 9]])
        array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        """
    for x in range(len(a)): 
        print (F[x]) 
    F = np.concatenate(F, axis=1)
    F = np.concatenate((X,F,F**2), axis=1)
        
    if scale is None:    
        scale = StandardScaler() 
        '''
        StandardScaler() will normalize the features (each column of X, INDIVIDUALLY !!!) 
        so that each column/feature/variable will have mean = 0 and standard deviation = 1.
        '''
        F = scale.fit_transform(F)
        '''
        fit() just calculates the parameters (e.g. μ and σ in case of StandardScaler) and saves them as an internal objects state. 
        Afterwards, you can call its transform() method to apply the transformation to a particular set of examples.
        fit_transform() oins these two steps and is used for the initial fitting of parameters on the training set x, 
        but it also returns a transformed x′. 
        '''
        return F, scale
    else:
        F = scale.transform(F)
        '''
        https://datascience.stackexchange.com/questions/12321/difference-between-fit-and-fit-transform-in-scikit-learn-models
        '''
        return F


#%%########### Initialize ####################################################
cols = ['HandStart','FirstDigitTouch',
        'BothStartLoadPhase','LiftOff',
        'Replace','BothReleased']

subjects = range(1,2)
idx_tot = []
scores_tot = []

###loop on subjects and 8 series for train data + 2 series for test data
for subject in subjects:

    X_train, y = load_data(subject)
    X_test, idx = load_data(subject,[9,10],'test')

################ Train classifiers ###########################################
    #lda = LDA()
    lr = LogisticRegression()
    """
    Just making instance
    For LogisticRegression: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    """
    
    X_train, scaler = compute_features(X_train)
    X_test = compute_features(X_test, scaler)   #pass the learned mean and std to normalized test data
    
    y = np.concatenate(y,axis=0)
    scores = np.empty((X_test.shape[0],6))
    
    downsample = 50
    for i in range(6):
        print('Train subject %d, class %s' % (subject, cols[i]))
        lr.fit(X_train[::downsample,:], y[::downsample,i])
        #lda.fit(X_train[::downsample,:], y[::downsample,i])
       
        scores[:,i] = 1*lr.predict_proba(X_test)[:,1]# + 0.5*lda.predict_proba(X_test)[:,1] # to predict probability
    scores_tot.append(scores)
    idx_tot.append(np.concatenate(idx))
    
#%%########### submission file ################################################
submission_file = 'Submission.csv'
# create pandas object for submission
submission = pd.DataFrame(index=np.concatenate(idx_tot),
                          columns=cols,
                          data=np.concatenate(scores_tot))

# write file
submission.to_csv(submission_file,index_label='id',float_format='%.3f')