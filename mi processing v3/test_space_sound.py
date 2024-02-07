import numpy as np
import pyaudio
import threading
from threading import Thread
import time
import math

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
        super().__init__()
        self.shared_state = shared_state
        self.stop_event = threading.Event()
        self.fs = 44100
        self.last_position = np.array([0, 0, 0], dtype=np.float64)
        self.listener_position = np.array([0, 0, 0], dtype=np.float64)
        self.left_ear_offset = np.array([-6, 0, 0], dtype=np.float64) 
        self.right_ear_offset = np.array([6, 0, 0], dtype=np.float64)  

    def calculate_amplitude(self, position, ear_position):
        distance = np.linalg.norm(position - ear_position)
        # Ajuste para uma escala perceptualmente uniforme
        max_distance = np.sqrt(10) * 5
        perceived_distance = (distance / max_distance) ** 0.2  # Exponente ajustável para afinar a percepção
        return map_value(perceived_distance, 0, 1, 1, 0)  # Mapeia a percepção linearizada para amplitude
   
    def sound_worker(self):
        while not self.stop_event.is_set():
            position = self.shared_state.get_position()
            if not np.array_equal(position, self.last_position):
                left_ear_position = self.listener_position + self.left_ear_offset
                right_ear_position = self.listener_position + self.right_ear_offset

                amplitude_left = self.calculate_amplitude(position, left_ear_position)
                amplitude_right = self.calculate_amplitude(position, right_ear_position)

                frequency = map_value(position[1], -10, 10, 100, 200)
                tone = generate_tone(frequency, 0.12, self.fs)

                # Combina as ondas para dois canais
                combined_tone = np.zeros((2, len(tone)))
                combined_tone[0] = tone * amplitude_left
                combined_tone[1] = tone * amplitude_right
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

    def update_position(self, new_position):
        with self.lock:
            # Restringindo o movimento dentro de um cubo de 20x20x20
            self.position = np.clip(new_position, -10, 10)

# Simulação do movimento browniano
def brownian_motion_simulation_3d(shared_state, steps, step_size):
    for _ in range(steps):
        movement = np.random.normal(0, step_size, 3)
        new_position = shared_state.get_position() + movement
        print(new_position)
        shared_state.update_position(new_position)
        time.sleep(0.1)

# Simulação do movimento browniano em 2D
def brownian_motion_simulation_2d(shared_state, steps, step_size):
    for _ in range(steps):
        current_position = shared_state.get_position()
        # Gera movimento apenas nas direções x e y
        movement = np.random.normal(0, step_size, 2)
        new_position = current_position + np.append(movement, 0)  # Adiciona 0 para a coordenada z
        print(new_position)
        shared_state.update_position(new_position)
        time.sleep(0.01)
        
# Função para simular movimento circular
def circular_motion_simulation(shared_state, steps, angle_step):
    radius = 10 # Raio do círculo
    angle = 0  # Ângulo inicial

    for _ in range(steps):
        # Coordenadas polares para cartesianas
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        shared_state.update_position(np.array([x, y, 0]))  # Z fixo em 0 para movimento 2D

        angle += angle_step  # Incrementa o ângulo
        if angle >= 2 * math.pi:
            angle -= 2 * math.pi  # Mantém o ângulo dentro do intervalo [0, 2π]
        time.sleep(0.01)  # Intervalo entre as atualizações de posição

# Exemplo de uso
shared_state = SharedState()
audio_thread = AudioThread(shared_state)
audio_thread.start()

try:
    # brownian_motion_simulation_2d(shared_state, 3000, 0.3)
    circular_motion_simulation(shared_state, 3600, math.pi/180)  
finally:
    audio_thread.stop()
