'''

Versão04: Recebe uma entrada do OV, processa e envia uma saída pro OV

'''

import numpy as np
import pickle
import configparser 
from pyriemann.estimation          import Covariances
from pyriemann.tangentspace        import tangent_space

class MyOVBox(OVBox):
    def __init__(self):
        OVBox.__init__(self)

        self.startTime           = 0.
        self.endTime             = 0.
        self.timeBuffer          = list()

        # par. dos dados de entrada
        self.nChannelIN          = 0
        self.samplingIN          = 1
        self.epochSampleCountIN  = 0
        self.samplePeriodIN      = .0
        self.dimensionSizesIN    = list()
        self.dimensionLabelsIN   = list()
        self.npBufferIN          = list()
        self.signalBufferIN      = None
        self.signalHeaderIN      = None
        self.samplePeriodIN      = .0

        # par. dos dados de saída
        self.nChannelOUT         = 0
        self.samplingOUT         = 1
        self.epochSampleCountOUT = 0
        self.dimensionSizesOUT   = list()
        self.dimensionLabelsOUT  = list()
        self.npBufferOUT         = list()
        self.signalBufferOUT     = None
        self.signalHeaderOUT     = None
        self.samplePeriodOUT     = .0

        # contador de processamentos
        self.nProc               = 0
        
        configParser = configparser.RawConfigParser()
        configFilePath  = r'C:/Users/Laboratorio/Documents/GitHub/real-time-bmi/mi processing v3/config.txt'
        configParser.read(configFilePath)

        Experiment      = configParser['PARAMETERS']['Experiment']
        Participant     = configParser['PARAMETERS']['Participant']
        type_classes    = configParser['PARAMETERS']['type_classes']
        Session_nb      = configParser['PARAMETERS']['Session_nb']
        Path_Save       = configParser['PARAMETERS']['Path_Save']
        sess_filename   = Path_Save + Participant + '/' + Experiment + '_' + type_classes +  '_' + Participant + '_Sess' + Session_nb
        # class_filename  = sess_filename + '_classifier.pkl'
        c_mean_filename = sess_filename + '_best_c_mean.pkl'
        pca_filename    = sess_filename + '_pca.pkl'
        
        with open(c_mean_filename, 'rb') as f: 
            self.best_c_mean = pickle.load(f)
        # with open(class_filename, 'rb') as f: 
        #     self.clf = pickle.load(f)
        with open(pca_filename,   'rb') as file:
            self.pca = pickle.load(file)

    def initialize(self):
        # definido o signalBufferIN pra não dar pau nas primeiras iterações de self.process()
        self.signalBufferIN      = np.zeros((1, 1))
        # definido o signalsignalHeaderIN pra não dar pau nas primeiras iterações de self.process()
        self.signalHeaderIN      = OVSignalHeader(0., 0., [1, 1], [1, 1], 1)
        pass

    def getInfosIN(self):
        self.samplingIN          = self.signalHeaderIN.samplingRate
        self.nChannelIN          = self.signalHeaderIN.dimensionSizes[0]
        self.epochSampleCountIN  = self.signalHeaderIN.dimensionSizes[1]
        self.dimensionSizesIN    = [self.nChannelIN, self.epochSampleCountIN]
        self.startTime           = self.signalHeaderIN.startTime
        self.endTime             = self.signalHeaderIN.endTime
        self.samplePeriodIN      = 1.*self.epochSampleCountIN / self.samplingIN
        self.dimensionLabelsIN   = self.signalHeaderIN.dimensionLabels
        pass

    def getDataIN(self):
        self.npBufferIN = np.array(self.signalBufferIN).reshape(tuple(self.dimensionSizesIN))
        pass

    def updateStartTime(self):
        self.startTime += self.samplePeriodIN
        pass

    def updateEndTime(self):
        self.endTime = float(self.startTime + self.samplePeriodIN)
        pass

    def updateTimeBuffer(self):
        self.timeBuffer = np.arange(self.startTime, self.endTime, 1./self.samplingIN)

    def makeInfosOUT(self):
        self.nChannelOUT         = 3
        self.epochSampleCountOUT = 1
        self.samplingOUT         = self.samplingIN

        for i in range(self.nChannelOUT):
            self.dimensionLabelsOUT.append('Canal'+str(i))
        self.dimensionLabelsOUT += self.epochSampleCountOUT*['']
        self.dimensionSizesOUT   = [self.nChannelOUT, self.epochSampleCountOUT]
        self.signalHeaderOUT     = OVSignalHeader(0., 0., self.dimensionSizesOUT, self.dimensionLabelsOUT, self.samplingOUT)
        self.output[0].append(self.signalHeaderOUT)
        pass

    def makeProcess(self):
        # Processar o dado como quiser
        cont = 0

        # saída = quadrado da entrada
        self.npBufferOUT = self.npBufferIN[np.newaxis, ...]
        print(self.npBufferOUT.shape)
        cov              = Covariances('oas').transform(self.npBufferOUT)
        tan_space_cov    = tangent_space(cov, self.best_c_mean)
        
        # Formata cabeçalho do dado de saída e envia(uso de nProc pra mandar só 1x )
        if self.nProc < 1:
            self.makeInfosOUT()
            self.nProc += 1

        # # Formata buffer de saída e envia
        start      = self.timeBuffer[0]
        end        = self.timeBuffer[-1] + 1./self.samplingOUT
        X_pca      = self.pca.transform(tan_space_cov)
        # prediction = self.clf.predict(X_pca)
        
        out = [X_pca[0][0], X_pca[0][1],1]
        # print(out)
        
        self.output[0].append(OVSignalBuffer(start, end, out))
        

    def process(self):

        for chunkIdx in range(len(self.input[0])):
            if type(self.input[0][chunkIdx]) == OVSignalHeader:
                self.signalHeaderIN = self.input[0].pop()
                # extrair características do cabeçalho
                self.getInfosIN()
            elif type(self.input[0][chunkIdx]) == OVSignalBuffer:
                self.signalBufferIN = self.input[0].pop()
                # converte buffer de dados em um np.array
                self.getDataIN()
                # processa dados, cria cabeçalho e envia para o OV
                self.makeProcess()

            elif type(self.input[0][chunkIdx]) == OVSignalEnd:
                self.output[0].append(self.input[0].pop())

        self.updateStartTime()
        self.updateEndTime()
        self.updateTimeBuffer()

        pass

box = MyOVBox()
