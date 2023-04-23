'''

Versão04: Recebe uma entrada do OV, processa e envia uma saída pro OV

'''


import numpy as np
import pickle
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
       
        with open('C:\\Users\\Laboratorio\\Documents\\GitHub\\real-time-bmi\\best_c_mean.pkl', 'rb') as f: 
            self.best_c_mean = pickle.load(f)
        
        with open('C:\\Users\\Laboratorio\\Documents\\GitHub\\real-time-bmi\\lda.pkl', 'rb') as f: 
            self.lda = pickle.load(f)

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
        self.nChannelOUT         = 2
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
        # for rowIndex, rowValue in enumerate(self.npBufferIN):
        #     # O for vai iterar 1x pra cada canal
        #     # rowIndex -> Nº Canal
        #     # rowValue -> lista de valores de cada row -> Mesmo q: self.npBufferIN[rowIndex]
        #     cont +=1
        #     # self.npBufferOUT[rowIndex, :] = self.npBufferIN[rowIndex, :]**2
        #     # if np.array_equal(self.npBufferIN[rowIndex],row):
        #     #     print("ok")
        #     pass

        # saída = quadrado da entrada
        self.npBufferOUT = self.npBufferIN[np.newaxis, ...]
        cov              = Covariances('oas').transform(self.npBufferOUT)
        tan_space_cov    = tangent_space(cov, self.best_c_mean)
        
        # Formata cabeçalho do dado de saída e envia(uso de nProc pra mandar só 1x )
        if self.nProc < 1:
            self.makeInfosOUT()
            self.nProc += 1

        # Formata buffer de saída e envia
        start = self.timeBuffer[0]
        end   = self.timeBuffer[-1] + 1./self.samplingOUT
        
        bufferElements = self.lda.predict_proba(tan_space_cov)[0]
        output_predict = [bufferElements[-1] - bufferElements[0]]
        print(output_predict)
        
        if output_predict[0] > 0:
          out = [0, bufferElements[-1]]
        elif output_predict[0] < 0:
          out = [bufferElements[0], 0]
        elif output_predict[0] == 0:
           out = [0, 0]
        
        self.output[0].append(OVSignalBuffer(start, end, out))

        pass

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
