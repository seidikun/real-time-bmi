import numpy as np
import pickle
import configparser
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import tangent_space
import matplotlib.pyplot as plt
import random

class MyOVBox(OVBox):
    def __init__(self):
        OVBox.__init__(self)

        # Carregar configurações e modelos
        configParser    = configparser.RawConfigParser()
        configFilePath  = r'C:\Users\seidi\Documents\GitHub\real-time-bmi\mi processing v2\config.txt'
        configParser.read(configFilePath)
        Experiment      = configParser['PARAMETERS']['Experiment']
        Participant     = configParser['PARAMETERS']['Participant']
        Session_nb      = configParser['PARAMETERS']['Session_nb']
        Path_Save       = configParser['PARAMETERS']['Path_Save']
        sess_filename   = Path_Save + Participant + '/' + Experiment + '_' + Participant + '_Sess' + Session_nb
        
        c_mean_filename = sess_filename + '_best_c_mean.pkl'
        class_filename  = sess_filename + '_classifier.pkl'
        c_mean_filename = sess_filename + '_best_c_mean.pkl'
        kde0_filename   = sess_filename + '_kde0.pkl'
        kde1_filename   = sess_filename + '_kde1.pkl'
        pca_filename    = sess_filename + '_pca.pkl'
        
        with open(c_mean_filename, 'rb') as f: 
            self.best_c_mean = pickle.load(f)
        with open(class_filename,  'rb') as f: 
            self.lda = pickle.load(f)
        with open(class_filename,  'rb') as file:
             self.clf = pickle.load(file)
        with open(pca_filename,   'rb') as file:
            self.pca = pickle.load(file)
        with open(kde0_filename,  'rb') as file:
            self.kde0 = pickle.load(file)
        with open(kde1_filename,  'rb') as file:
            self.kde1 = pickle.load(file)
             
        # Definindo o espaço de visualização
        x_min, x_max = -3, 3  # Ajuste estes valores conforme necessário
        y_min, y_max = -3, 3  # Ajuste estes valores conforme necessário
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

        # Calculando as densidades no grid
        zz0 = self.kde0(np.vstack([xx.ravel(), yy.ravel()]))
        zz1 = self.kde1(np.vstack([xx.ravel(), yy.ravel()]))

        # Configurações iniciais do gráfico
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(-3.1, 3.1)
        self.ax.set_ylim(-3.1, 3.1)
        self.scatter_plot, = self.ax.plot([], [], 'o')
        
        # Plotando as curvas de densidade no eixo especificado
        self.ax.contourf(xx, yy, zz0.reshape(xx.shape), alpha=0.5, cmap='Blues')
        self.ax.contourf(xx, yy, zz1.reshape(xx.shape), alpha=0.5, cmap='Reds')
        self.ax.set_title('KDE Contours')
        

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
        
        bufferElements = self.clf.predict_proba(tan_space_cov)[0]
        out = [max(bufferElements), 0] if bufferElements[-1] > bufferElements[0] else [0, max(bufferElements)]
        
        self.output[0].append(OVSignalBuffer(self.startTime, self.endTime, out))
        
        self.update_plot()

        self.startTime += 1.0 * self.epochSampleCountIN / self.samplingIN
        self.endTime    = self.startTime + 1.0 * self.epochSampleCountIN / self.samplingIN

    def update_plot(self):
        self.scatter_plot.set_data(random.random() * 10, random.random() * 10)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.00001)

box = MyOVBox()
