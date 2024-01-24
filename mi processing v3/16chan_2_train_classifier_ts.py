import pickle
import random
import numpy  as np
import pandas as pd
import configparser 
from scipy                  import signal
from sklearn.metrics        import cohen_kappa_score
from sklearn.svm            import SVC
from sklearn.decomposition  import PCA
from scipy.stats            import gaussian_kde
from pyriemann.estimation   import Covariances
from pyriemann.tangentspace import tangent_space
from pyriemann.utils.mean   import mean_covariance
from matplotlib             import pyplot as plt

# Read config file
configParser    = configparser.RawConfigParser()   
configFilePath  = r'.\config.txt'
configParser.read(configFilePath)
Experiment      = configParser['PARAMETERS']['Experiment']
Participant     = configParser['PARAMETERS']['Participant']
Session_nb      = configParser['PARAMETERS']['Session_nb']
Path_Project    = configParser['PARAMETERS']['Path_Project']
Path_Save       = configParser['PARAMETERS']['Path_Save']
sess_filename   = Path_Save + Participant + '/' + Experiment + '_' + Participant + '_Sess' + Session_nb

sess_filename =  'C:/Users/seidi/Desktop/Data/EG102/motor_mi_EG102_online_6d'
filename        = sess_filename + '.csv'
class_filename  = sess_filename + '_classifier.pkl'
c_mean_filename = sess_filename + '_best_c_mean.pkl'
kde0_filename   = sess_filename + '_kde0.pkl'
kde1_filename   = sess_filename + '_kde1.pkl'
pca_filename    = sess_filename + '_pca.pkl'

# ------------------------------ PRE-PROCESSING ------------------------------#
# Parameters
overlap_step    = 0.0625  # Temporal epoch overlap
max_trial_t     = 3.75 # Trial nmax duration
b, a            = signal.butter(4, (8,30), 'bandpass', fs=512)
left_label_     = 0
right_label_    = 1

# Read .csv file
df_me           = pd.read_csv(filename)

# Get indexes of left MI and right MI
event_id        = pd.to_numeric(df_me['Event Id'], errors='coerce').fillna(0).astype(np.int64)
left_inds       = event_id.index[event_id == 769].tolist()
right_inds      = event_id.index[event_id == 770].tolist()
  
# Sample frequency
Fs              = 1/np.mean(np.diff(df_me['Time:512Hz']))

# Get EEG data only and band-pass filter
data_eeg        = df_me.iloc[:, 2:18].values
data_eeg        = signal.filtfilt(b, a, data_eeg.T).T
    
# Time vector of epoch
trange          = np.arange(int(Fs*0.5),int(Fs*2.5)) 
  
# Windowing
while trange[-1] < int(max_trial_t*Fs):        

  # Left MI epochs
  ts            = [i + trange for i in left_inds]
  epochs_left   = data_eeg[ts,:]
  labels        = [left_label_ for i in left_inds]
  trial_num     = [i for i in range(len(left_inds))]

  # Right MI epochs
  ts            = [i + trange for i in right_inds]
  epochs_right  = data_eeg[ts,:]
  epochs_mne    = np.vstack([epochs_left, epochs_right])
  labels        = np.append(labels,    [right_label_ for i in right_inds])
  trial_num     = np.append(trial_num, [i for i in range(len(right_inds))])

  # Create epochs stack, labels array and trial id array
  if 'data' not in locals():
    data          = epochs_mne
    all_labels    = labels
    all_trial_num = trial_num
  else:
    data          = np.vstack([data, epochs_mne])
    all_labels    = np.append(all_labels, labels)
    all_trial_num = np.append(all_trial_num, trial_num)

  # Slide window
  trange += int(overlap_step*Fs)

# Trials randomization
p             = np.random.permutation(len(all_labels))
data          = np.swapaxes(data[p,:,:,],1,2)
all_labels    = all_labels[p]
all_trial_num = all_trial_num[p]

# Unique trial id
unique_labels = np.unique(all_trial_num)

# ------------------------------ MODEL VALIDATION ----------------------------#
# Init variables
best_accuracy = 0
best_kappa    = 0
best_c_mean   = None
n_folds       = 20

# Init metrics arrays
accuracies    = np.zeros(n_folds)
kappas        = np.zeros(n_folds)

# Cross-validation
for fold in range(n_folds):

    # Randomly choose 2 trial ids for validation
    val_labels       = random.sample(list(unique_labels), 2)

    # Trials indexes
    train_inds       = np.where(np.logical_not(np.isin(all_trial_num, val_labels)))[0]
    val_inds         = np.where(np.isin(all_trial_num, val_labels))[0]

    # Transform to covariance matrices
    cov_data_train   = Covariances('oas').transform(data[train_inds])
    cov_data_val     = Covariances('oas').transform(data[val_inds])
    
    # Train-validation split
    X_train, y_train = cov_data_train, all_labels[train_inds]
    X_val, y_val     = cov_data_val, all_labels[val_inds]
    
    # Mean covariance matrix
    C_mean           = mean_covariance(X_train)
    
    # Project matrices to tangential space
    tan_space_train  = tangent_space(X_train, C_mean)
    tan_space_val    = tangent_space(X_val, C_mean)
    
    # Train classifier
    classifier = SVC(random_state=0)
    classifier.fit(tan_space_train, y_train)
    
    # Metrics
    print('\nFold', fold+1)
    accuracy         = classifier.score(tan_space_val, y_val)
    print('Accuracy:', accuracy)
    
    y_pred_val       = classifier.predict(tan_space_val)
    kappa_val        = cohen_kappa_score(y_val, y_pred_val)
    print('Kappa:   ', kappa_val)
    
    accuracies[fold] = accuracy
    kappas[fold]     = kappa_val

    # Select best c mean according to kappa
    if kappa_val > best_kappa:
        best_kappa  = kappa_val
        best_c_mean = C_mean

# Calculate mean accuracy
print("\nMean cross-validation accuracy:", np.mean(accuracies))
print("Mean cross-validation kappa:   ", np.mean(kappas))

# Saving the objects:
with open(c_mean_filename, 'wb') as f: 
    pickle.dump(best_c_mean, f)

# ------------------------------ MODEL TRAINING ------------------------------#
# Transform epochs into covariance matrices
cov_data_all  = Covariances('oas').transform(data)

# Project cov matrices to tangential space trhough best_c_mean
tan_space_all = tangent_space(cov_data_all, best_c_mean) 


# ------------------------------ MENTAL MAPPING ------------------------------#
# 3. Aplicando PCA para reduzir para duas dimensões (para visualização)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(tan_space_all)

# Train classifier
clf = SVC(kernel='linear', decision_function_shape='ovo', probability = True)
clf.fit(X_pca, all_labels)


# 4. Preparando para desenhar as distribuições de KDE
x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))

# Função para calcular a densidade KDE
def kde_scipy(x, y, data):
    kde = gaussian_kde(data.T)
    return kde.evaluate(np.vstack([x.ravel(), y.ravel()])).reshape(x.shape)

# 5. Aplicando KDE para cada classe
colors = ['Reds', 'Blues', 'Greens', 'Purples']
for i in [left_label_, right_label_]:
    zi = kde_scipy(xx, yy, X_pca[all_labels == i])
    plt.contourf(xx, yy, zi, alpha=0.5, cmap=colors[i], levels=np.linspace(zi.min(), zi.max(), 7))

# 6. Desenhando a fronteira de decisão do SVC
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])

# 7. Desenhando os pontos de dados
#plt.scatter(X_pca[:, 0], X_pca[:, 1], c=all_labels, s=30, cmap='viridis', edgecolors='k')

plt.xlabel('PCA Feature 1')
plt.ylabel('PCA Feature 2')
plt.title('SVC Decision Boundary with KDE for 2 Separated Classes')
plt.show()

# Calculando a densidade KDE
kde0 = gaussian_kde(X_pca[all_labels == left_label_].T)
kde1 = gaussian_kde(X_pca[all_labels == right_label_].T)

with open(class_filename, 'wb') as file:
    pickle.dump(clf, file)
with open(pca_filename, 'wb') as file:
    pickle.dump(pca, file)
with open(kde0_filename, 'wb') as file:
    pickle.dump(kde0, file)
with open(kde1_filename, 'wb') as file:
    pickle.dump(kde1, file)