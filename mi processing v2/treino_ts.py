# Instalação de pacotes e imports
import pickle
import random
import numpy  as np
import pandas as pd
from scipy                         import signal
from sklearn.metrics               import cohen_kappa_score
from sklearn.linear_model          import LogisticRegression
from pyriemann.estimation          import Covariances
from pyriemann.tangentspace        import tangent_space
from pyriemann.utils.mean          import mean_covariance

########################## CHANGE THE NAMES HERE ###########################
path_files      = 'C:\\Users\\seidi\\Documents\\GitHub\\real-time-bmi\\mi processing v2\\'
path_data       = 'C:\\Users\\seidi\\Documents\\GitHub\\real-time-bmi\\test_data\\'
data_filename   = 'record-[2023.04.24-12.02.42]-seidi-mi-hands.csv'
c_mean_filename = 'best_c_mean.pkl'
class_filename  = 'lda.pkl'
nb_chans        = 16
############################################################################

# --------------------------- Pre-processing --------------------------------
# Read data file
df_me        = pd.read_csv(path_data + data_filename)
Fs           = 1/np.mean(np.diff(df_me['Time:512Hz']))

# Right MI and Lef MI indexes
event_id     = pd.to_numeric(df_me['Event Id'], errors='coerce').fillna(0).astype(np.int64)
left_inds    = event_id.index[event_id == 769].tolist()
right_inds   = event_id.index[event_id == 770].tolist()
  
# Signal filtering
b, a         = signal.butter(4, (8,30), 'bandpass', fs=Fs)
data_eeg     = df_me.iloc[:, 2:2 + nb_chans].values
data_eeg     = signal.filtfilt(b, a, data_eeg.T).T

# Epoching parameters
trange       = np.arange(int(Fs*0.5),int(Fs*2.5), 4) # 0.5s to 2.5s window with downsample by 4
overlap_step = 0.0625
max_trial_t  = 3.75 
    
# Time epoching
while trange[-1] < int(max_trial_t*Fs):        

  # Left MI epoching
  ts            = [i + trange for i in left_inds]
  epochs_left   = data_eeg[ts,:]
  labels        = [0 for i in left_inds]
  trial_num     = [i for i in range(len(left_inds))]

  # Right MI epoching
  ts            = [i + trange for i in right_inds]
  epochs_right  = data_eeg[ts,:]
  labels        = np.append(labels, [1 for i in right_inds])
  trial_num     = np.append(trial_num, [i for i in range(len(right_inds))])
  
  epochs_mne    = np.vstack([epochs_left, epochs_right])

  # Concatenate epochs
  if 'data' not in locals():
    data          = epochs_mne
    all_labels    = labels
    all_trial_num = trial_num
  else:
    data          = np.vstack([data, epochs_mne])
    all_labels    = np.append(all_labels, labels)
    all_trial_num = np.append(all_trial_num, trial_num)

  # New time window
  trange += int(overlap_step*Fs)

# Randomize epochs
p             = np.random.permutation(len(all_labels))
data          = np.swapaxes(data[p,:,:,],1,2)
all_labels    = all_labels[p]
all_trial_num = all_trial_num[p]

# -------------------------- Cross-Validation -------------------------------
# Init metrics and c_mean
best_accuracy, best_kappa = 0, 0
best_c_mean   = None

# Number of cross-val folds
n_folds       = 20
accuracies    = np.zeros(n_folds)
kappas        = np.zeros(n_folds)

# Each unique trial id
unique_labels = np.unique(all_trial_num)

# Fazer a validação cruzada
for fold in range(n_folds):

    # Choose randomly two trial ids
    val_labels           = random.sample(list(unique_labels), 2)

    # train and validation split indexes
    train_inds           = np.where(np.logical_not(np.isin(all_trial_num, val_labels)))[0]
    val_inds             = np.where(np.isin(all_trial_num, val_labels))[0]

    # Get covariance matrices from epochs
    cov_data_train       = Covariances().transform(data[train_inds])
    cov_data_val         = Covariances().transform(data[val_inds])
    
    # Split data in train and validation
    X_train, y_train     = cov_data_train, all_labels[train_inds]
    X_val, y_val         = cov_data_val, all_labels[val_inds]
    
    # Calculate mean covariance matrix
    C_mean               = mean_covariance(X_train)
    
    # Tangential plane projection by C_mean
    tan_space_covs_train = tangent_space(X_train, C_mean)
    tan_space_covs_val   = tangent_space(X_val, C_mean)
    
    # Logistic Regression train of tan_space feature vectors
    lr = LogisticRegression(random_state=0)
    lr.fit(tan_space_covs_train, y_train)
    
    # Metrics
    accuracy = lr.score(tan_space_covs_val, y_val)
    print('\n\nFold', fold + 1)
    print("  accuracy: {:.2f}".format(accuracy))
    
    y_pred_val = lr.predict(tan_space_covs_val)
    kappa_val  = cohen_kappa_score(y_val, y_pred_val)
    print("  kappa:    {:.2f}".format(kappa_val), end = "")
    
    accuracies[fold] = accuracy
    kappas[fold]     = kappa_val

    # Save best C_mean
    if kappa_val > best_kappa:
        print('  NEW BEST KAPPA!', end = "")
        best_kappa  = kappa_val
        best_c_mean = C_mean

# Calculate mean accuracy
mean_accuracy = np.mean(accuracies)
print("\n\nCross-Validation mean accuracy: {:.2f}".format(mean_accuracy))

# Saving the objects:
with open(path_files + c_mean_filename, 'wb') as f: 
    pickle.dump(best_c_mean, f)

# ------------------------- Classifier training ------------------------------
# Get covariance matrices
cov_data_train       = Covariances().transform(data)

# Tangential plane projection by best_c_mean
tan_space_covs_train = tangent_space(cov_data_train, best_c_mean)

# Fit logistic regression
clf = LogisticRegression().fit(tan_space_covs_train, all_labels)

# Save to file in the current working directory
with open(path_files + class_filename, 'wb') as file:
    pickle.dump(clf, file)