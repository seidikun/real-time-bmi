import numpy as np
import pyaudio
import threading
from threading import Thread
import time
import paho.mqtt.client as mqtt
import configparser
import pickle

configParser    = configparser.RawConfigParser()
configFilePath  = r'C:/Users/Laboratorio/Documents/GitHub/real-time-bmi/mi processing v3/config.txt'
configParser.read(configFilePath)
Experiment      = configParser['PARAMETERS']['Experiment']
type_classes    = configParser['PARAMETERS']['type_classes']
Participant     = configParser['PARAMETERS']['Participant']
Session_nb      = configParser['PARAMETERS']['Session_nb']
Path_Save       = configParser['PARAMETERS']['Path_Save']
sess_filename   = Path_Save + Participant + '/' + Experiment + '_' + type_classes + '_' + Participant + '_Sess' + Session_nb

# Defina as configurações do broker
broker_address  = "localhost" 
port            = 1883  
topic           = "PCA_values"


with open(sess_filename + '_range_pca.pkl', 'rb') as file:
    x_min, x_max, y_min, y_max = pickle.load(file)

# Função para gerar um tom
def generate_tone(frequency, duration, sample_rate):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    tone = np.sin(2 * np.pi * frequency * t)

    # Aplicando fade in e fade out
    fade_length          = int(sample_rate * duration * 0.05)  # 5% da duração total do tom
    fade_in              = np.linspace(0, 1, fade_length)
    fade_out             = np.linspace(1, 0, fade_length)
    tone[:fade_length]  *= fade_in
    tone[-fade_length:] *= fade_out
    return tone

# Função para mapear valores
def map_value(value, min_input, max_input, min_output, max_output):
    return (value - min_input) / (max_input - min_input) * (max_output - min_output) + min_output

# Classe para controlar o áudio
class AudioThread(threading.Thread):
    def __init__(self, shared_state):
        global x_min, x_max, y_min, y_max
        self.center_x = x_min + (x_max - x_min)/2
        self.center_y = y_min + (y_max - y_min)/2
        super().__init__()
        self.shared_state = shared_state
        self.stop_event = threading.Event()
        self.fs                = 44100
        self.last_position     = np.array([0, 0, 0], dtype=np.float64)
        self.listener_position = np.array([self.center_x, self.center_y, 0], dtype=np.float64)
        self.left_ear_offset   = np.array([self.center_x-3, self.center_y, 0], dtype=np.float64) 
        self.right_ear_offset  = np.array([self.center_x+3, self.center_y, 0], dtype=np.float64)  

    def calculate_amplitude(self, position, ear_position):
        distance = np.linalg.norm(position - ear_position)
        # Ajuste para uma escala perceptualmente uniforme
        max_distance = np.sqrt(self.center_x**2 + self.center_y**2)
        perceived_distance = (distance / max_distance) ** 0.2  # Exponente ajustável para afinar a percepção
        return map_value(perceived_distance, 0, 1, 1, 0)  # Mapeia a percepção linearizada para amplitude
   
    def sound_worker(self):
        while not self.stop_event.is_set():
            position = self.shared_state.get_position()
            if not np.array_equal(position, self.last_position):
                left_ear_position  = self.listener_position + self.left_ear_offset
                right_ear_position = self.listener_position + self.right_ear_offset

                amplitude_left     = self.calculate_amplitude(position, left_ear_position)
                amplitude_right    = self.calculate_amplitude(position, right_ear_position)

                frequency = map_value(position[1], y_min, y_max, 50, 400)
                tone = generate_tone(frequency, 0.12, self.fs)
                print(frequency, amplitude_left, amplitude_right)

                # Combina as ondas para dois canais
                combined_tone = np.zeros((2, len(tone)))
                combined_tone[1] = tone * amplitude_left
                combined_tone[0] = tone * amplitude_right
                self.stream.write(combined_tone.T.astype(np.float32).tobytes())
                self.last_position = position

    def start(self):
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=pyaudio.paFloat32, channels=2, rate=self.fs, output=True)
        self.sound_thread = Thread(target=self.sound_worker)
        self.sound_thread.start()
        print('Audio inicializado')

    def stop(self):
        self.stop_event.set()
        self.sound_thread.join()
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

# Classe para simular o movimento browniano
class SharedState:
    def __init__(self):
        self.position = np.array([0, 0, 0], dtype=np.float64)
        self.lock = threading.Lock()

    def get_position(self):
        with self.lock:
            return self.position.copy()

    def update_position_from_mqtt(self, x, y):
        global x_min, x_max, y_min, y_max
        with self.lock:
            x_clipped = np.clip(x, x_min, x_max)
            y_clipped = np.clip(y, y_min, y_max)
            self.position = np.array([x_clipped, y_clipped, 0])  

# Função de callback chamada quando a conexão com o broker é estabelecida
def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))
    client.subscribe(topic)


# Função chamada quando uma mensagem é recebida do servidor MQTT.
def on_message(client, userdata, msg):
    global value_0, value_1
    message = msg.payload.decode()
    value_0, value_1, label = map(float, message.split(','))
    # print(value_0, value_1)

# Função para rodar o loop MQTT em um thread separado
def mqtt_thread():
    client            = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(broker_address, port, 60)
    client.loop_forever()
    
value_0, value_1 = 0,0

# Cria e inicia o thread MQTT
mqtt_thread        = threading.Thread(target=mqtt_thread)
mqtt_thread.daemon = True
mqtt_thread.start()

# Inicialização do sistema de áudio e estado compartilhado
shared_state = SharedState()
audio_thread = AudioThread(shared_state)
audio_thread.start()

try:
    while True:  # Loop para atualizar continuamente a posição
        shared_state.update_position_from_mqtt(value_0, value_1)
        time.sleep(0.01)  # Pode ajustar esse tempo conforme necessário
except KeyboardInterrupt:
    audio_thread.stop()