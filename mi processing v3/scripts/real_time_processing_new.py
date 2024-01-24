import numpy as np
import pickle
import configparser
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import tangent_space
import matplotlib.pyplot as plt
import random

from scipy.stats import gaussian_kde


class MyOVBox(OVBox):
    def __init__(self):
        OVBox.__init__(self)

        # Carregar configurações e modelos
        configParser    = configparser.RawConfigParser()
        configFilePath  = r'C:\Users\seidi\Documents\GitHub\real-time-bmi\mi processing v3\config.txt'
        configParser.read(configFilePath)
        Experiment      = configParser['PARAMETERS']['Experiment']
        Participant     = configParser['PARAMETERS']['Participant']
        Session_nb      = configParser['PARAMETERS']['Session_nb']
        Path_Save       = configParser['PARAMETERS']['Path_Save']
        sess_filename   = Path_Save + Participant + '/' + Experiment + '_' + Participant + '_Sess' + Session_nb
        
        sess_filename =  'C:/Users/seidi/Desktop/Data/EG102/motor_mi_EG102_online_6d'
        c_mean_filename = sess_filename + '_best_c_mean.pkl'
        class_filename  = sess_filename + '_classifier.pkl'
        c_mean_filename = sess_filename + '_best_c_mean.pkl'
        pca_filename    = sess_filename + '_pca.pkl'
        
        with open(c_mean_filename, 'rb') as f: 
            self.best_c_mean = pickle.load(f)
        with open(class_filename,  'rb') as f: 
            self.lda = pickle.load(f)
        with open(class_filename,  'rb') as file:
            self.clf = pickle.load(file)
        with open(pca_filename,   'rb') as file:
            self.pca = pickle.load(file)
    
    def process(self):
        for chunkIdx in range(len(self.input[0])):
            if type(self.input[0][chunkIdx]) == OVSignalHeader:
                self.signalHeaderIN = self.input[0].pop()
                self.initialize_signal(self.signalHeaderIN)
            elif type(self.input[0][chunkIdx]) == OVSignalBuffer:
                self.signalBufferIN = self.input[0].pop()
                self.makeProcess(self.signalBufferIN)
            elif type(self.input[0][chunkIdx]) == OVSignalEnd:
                self.output[0].append(self.input[0].pop())

    def initialize_signal(self, signalHeader):
        self.samplingIN         = signalHeader.samplingRate
        self.nChannelIN         = signalHeader.dimensionSizes[0]
        self.epochSampleCountIN = signalHeader.dimensionSizes[1]
        self.startTime          = signalHeader.startTime
        self.endTime            = signalHeader.endTime
        self.dimensionSizesIN   = [self.nChannelIN, self.epochSampleCountIN]

    def makeProcess(self, signalBuffer):
        npBufferIN             = np.array(signalBuffer).reshape(tuple(self.dimensionSizesIN))
        cov                    = Covariances('oas').transform(npBufferIN[np.newaxis, ...])
        tan_space_cov          = tangent_space(cov, self.best_c_mean)
        
        X_pca = self.pca.transform(tan_space_cov)
        print(X_pca)
        
        out = [[X_pca[0][0], X_pca[0][1]]]
        
        self.output[0].append(OVSignalBuffer(self.startTime, self.endTime, out))

        self.startTime += 1.0 * self.epochSampleCountIN / self.samplingIN
        self.endTime    = self.startTime + 1.0 * self.epochSampleCountIN / self.samplingIN


box = MyOVBox()
