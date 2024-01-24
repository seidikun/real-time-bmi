import numpy as np
import pandas as pd
from matplotlib             import pyplot as plt
from scipy                  import signal
from scipy.stats            import gaussian_kde
from sklearn.decomposition  import PCA
from sklearn                import svm
from pyriemann.estimation   import Covariances
from pyriemann.utils.mean   import mean_covariance
from pyriemann.tangentspace import tangent_space
import configparser
import pickle
import random

configParser    = configparser.RawConfigParser()
configFilePath  = r'C:\Users\Laboratorio\Documents\GitHub\real-time-bmi\mi processing v3\config.txt'
configParser.read(configFilePath)
Experiment      = configParser['PARAMETERS']['Experiment']
Participant     = configParser['PARAMETERS']['Participant']
Session_nb      = configParser['PARAMETERS']['Session_nb']
Path_Save       = configParser['PARAMETERS']['Path_Save']
sess_filename   = Path_Save + Participant + '/' + Experiment + '_4classes_' + Participant + '_Sess' + Session_nb

label_left_  = 0
label_right_ = 1
label_feet_  = 2
label_rest_  = 3
epoch_size   = 1 #s

# ParÂmetros do dado
Fs         = 512
b, a       = signal.butter(4, (8, 30), 'bandpass', fs=Fs)
columns    = ['Channel ' + str(i) for i in  range(1,17)]
filename   = sess_filename + '.csv'
df_me      = pd.read_csv(filename, low_memory=False)
event_id   = pd.to_numeric(df_me['Event Id'], errors='coerce').fillna(0).astype(np.int64)
left_inds  = event_id.index[event_id == 769].tolist()
right_inds = event_id.index[event_id == 770].tolist()
feet_inds  = event_id.index[event_id == 1089].tolist()
rest_inds  = event_id.index[event_id == 1090].tolist()

def data_preparation(data_eeg, a, b, columns, left_inds, right_inds, feet_inds, rest_inds , epoch_size):
  # Define parâmetros
  count_files  = 0    # contador de arquivos
  overlap_step = 0.0625  # tamanho do passo para sobreposição de janelas
  max_trial_t  = 3.75 # tamanho máximo de uma janela de tentativa

  # Define os índices das janelas
  trange        = np.arange(int(Fs*0.5),int(Fs*(0.5 + epoch_size)))

  # Loop pelas janelas
  while trange[-1] < int(max_trial_t*Fs):

    # Obtém as janelas de imagética motora da mão esquerda
    ts            = [i + trange for i in left_inds]
    epochs_left   = data_eeg[ts,:]
    labels        = [label_left_ for i in left_inds]
    trial_num     = [i for i in range(len(left_inds))]

    # Obtém as janelas de imagética motora da mão direita
    ts            = [i + trange for i in right_inds]
    epochs_right  = data_eeg[ts,:]
    epochs_mne    = np.vstack([epochs_left, epochs_right])
    labels        = np.append(labels, [label_right_ for i in right_inds])
    trial_num     = np.append(trial_num, [i for i in range(len(right_inds))])

    # Obtém as janelas de imagética motora da mão direita
    ts            = [i + trange for i in feet_inds]
    epochs_feet   = data_eeg[ts,:]
    epochs_mne    = np.vstack([epochs_mne, epochs_feet])
    labels        = np.append(labels, [label_feet_ for i in feet_inds])
    trial_num     = np.append(trial_num, [i for i in range(len(feet_inds))])

    # Obtém as janelas de imagética motora da mão direita
    ts            = [i + trange for i in rest_inds]
    epochs_rest   = data_eeg[ts,:]
    epochs_mne    = np.vstack([epochs_mne, epochs_rest])
    labels        = np.append(labels, [label_rest_ for i in rest_inds])
    trial_num     = np.append(trial_num, [i for i in range(len(rest_inds))])
    # Concatena as janelas de ambos os lados
    if count_files == 0:
      data          = epochs_mne
      all_labels    = labels
      all_trial_num = trial_num
    else:
      data          = np.vstack([data, epochs_mne])
      all_labels    = np.append(all_labels, labels)
      all_trial_num = np.append(all_trial_num, trial_num)

    # Atualiza o índice das janelas
    trange += int(overlap_step*Fs)

    # Atualiza o contador de arquivos
    count_files += 1

  # Aleatoriza as amostras
  p             = np.random.permutation(len(all_labels))
  data          = np.swapaxes(data[p,:,:,],1,2)
  all_labels    = all_labels[p]
  all_trial_num = all_trial_num[p]
  return data, all_labels, all_trial_num

def all_data_epoching(data_eeg, Fs, epoch_size):
  # Define parâmetros
  overlap_step = 0.01  # tamanho do passo para sobreposição de janelas
  trange       = np.arange(0,int(Fs*epoch_size))
  max_t        = data_eeg.shape[0]
  count_epoch  = 0
  data         = []
  
  # Loop pelas janelas
  while trange[-1] < max_t:
    epoch   = data_eeg[trange,:]
    data.append(epoch)

    # Atualiza o índice das janelas
    trange += int(overlap_step*Fs)
    count_epoch += 1

  np.array(data)

  data          = np.swapaxes(data,1,2)
  return data

# Janelando as tentativas IM
data_eeg                 = df_me[columns].to_numpy()
data_eeg                 = signal.filtfilt(b, a, data_eeg.T).T
data, labels, all_trial_num = data_preparation(data_eeg, a, b, columns, left_inds, right_inds,feet_inds, rest_inds , epoch_size)
unique_labels            = np.unique(all_trial_num)

# Init variables
best_acc      = 0
best_c_mean   = None
n_folds       = 20

# Init metrics arrays
accuracies    = np.zeros(n_folds)

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
    X_train, y_train = cov_data_train, labels[train_inds]
    X_val, y_val     = cov_data_val, labels[val_inds]
    
    # Mean covariance matrix
    C_mean           = mean_covariance(X_train)
    
    # Project matrices to tangential space
    tan_space_train  = tangent_space(X_train, C_mean)
    tan_space_val    = tangent_space(X_val, C_mean)
    

    # Treinando o classificador SVC
    classifier = svm.SVC(kernel='linear', C=1)
    classifier.fit(tan_space_train, y_train)
    
    # Metrics
    print('\nFold', fold+1)
    accuracy         = classifier.score(tan_space_val, y_val)
    print('Accuracy:', accuracy)
    
    y_pred_val       = classifier.predict(tan_space_val)
    
    accuracies[fold] = accuracy

    # Select best c mean according to kappa
    if accuracy > best_acc:
        best_acc  = accuracy
        best_c_mean = C_mean

# Calculate mean accuracy
print("\nMean cross-validation accuracy:", np.mean(accuracies))

# Saving the objects:
with open(sess_filename + '_best_c_mean.pkl', 'wb') as f: 
    pickle.dump(best_c_mean, f)

# Janelando o sinal todo
all_epochs               = all_data_epoching(data_eeg, Fs, epoch_size)
cov_data_all             = Covariances(estimator='lwf').transform(all_epochs)
tan_vectors_all          = tangent_space(cov_data_all, best_c_mean)

# Aplicando PCA para reduzir para duas dimensões   
cov_dat          = Covariances('oas').transform(data)
X                = cov_dat
tan_vectors_mi   = tangent_space(X, best_c_mean)
pca              = PCA(n_components=2)
X_pca            = pca.fit_transform(tan_vectors_mi)

# Treinando o classificador SVC
clf = svm.SVC(kernel='linear', C=1)
clf.fit(X_pca, labels)

# Calculando a densidade de KDE para cada classe
x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))

kde0 = gaussian_kde(X_pca[labels == label_left_].T)
kde1 = gaussian_kde(X_pca[labels == label_right_].T)
kde2 = gaussian_kde(X_pca[labels == label_feet_].T)
kde3 = gaussian_kde(X_pca[labels == label_rest_].T)

# Aplicando KDE para cada classe
z0 = kde0.evaluate(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
z1 = kde1.evaluate(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
z2 = kde1.evaluate(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
z3 = kde1.evaluate(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

# Plotando o resultado
plt.contourf(xx, yy, z0, alpha=0.5, levels=np.linspace(z0.min(), z0.max(), 7), cmap='Blues')
plt.contourf(xx, yy, z1, alpha=0.5, levels=np.linspace(z1.min(), z1.max(), 7), cmap='Reds')
plt.contourf(xx, yy, z2, alpha=0.5, levels=np.linspace(z1.min(), z1.max(), 7), cmap='Oranges')
plt.contourf(xx, yy, z3, alpha=0.5, levels=np.linspace(z1.min(), z1.max(), 7), cmap='Greys')

# Desenhando a fronteira de decisão do SVC
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, s=30, cmap=plt.cm.bwr, edgecolors='k')
plt.xlabel('PCA Feature 1')
plt.ylabel('PCA Feature 2')
plt.title('SVC Decision Boundary with PCA and KDE')
plt.show()

# Salvando o modelo do classificador e os dados de KDE
with open(sess_filename + '_classifier_4class.pkl', 'wb') as file:
    pickle.dump(clf, file)
with open(sess_filename + '_pca_4class.pkl', 'wb') as file:
    pickle.dump(pca, file)
with open(sess_filename + '_kde0_4class.pkl', 'wb') as file:
    pickle.dump(kde0, file)
with open(sess_filename + '_kde1_4class.pkl', 'wb') as file:
    pickle.dump(kde1, file)
with open(sess_filename + '_kde2_4class.pkl', 'wb') as file:
    pickle.dump(kde0, file)
with open(sess_filename + '_kde3_4class.pkl', 'wb') as file:
    pickle.dump(kde1, file)