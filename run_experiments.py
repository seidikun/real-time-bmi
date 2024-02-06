import subprocess
import os
import csv
from datetime import datetime

# Função para verificar se o registro já existe
def check_record_exists(filepath, dict_exp):
    exists = False
    try:
        with open(filepath, mode='r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if (row['Experiment']  == dict_exp['Experiment'] and
                    row['Participant'] == dict_exp['Participant'] and
                    row['Session']     == dict_exp['Session'] and
                    row['Run']         == dict_exp['Run']):
                    exists = True
                    break
    except FileNotFoundError:
        print(f"Arquivo {filepath} não encontrado.")
    except Exception as e:
        print(f"Erro ao ler o arquivo {filepath}: {e}")
    return exists

# Função para adicionar um novo registro
def add_record(filepath, search_data):
    try:
        with open(filepath, mode='a', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['Experiment', 'Participant', 'Session', 'Run', 'Filename', 'Timestamp', 'Computer']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(search_data)
            print("Novo registro adicionado com sucesso.")
    except Exception as e:
        print(f"Erro ao escrever no arquivo {filepath}: {e}")

# Variáveis globais
project_path           = os.path.dirname(os.path.realpath(__file__)).replace("\\", "/")
computer_name          = os.environ['COMPUTERNAME']

openvibe_designer_path = r'C:/Program Files/openvibe-3.5.0-64bit/openvibe-designer.cmd'
data_path              = 'C:/Users/seidi/Desktop/Data/'
experiments            = ['RIEMANN_LDA']
participantes          = ['SY100', 'TEST']

# Lista de experimentos disponíveis
print("Escolha um experimento:")
for i, exp in enumerate(experiments, 1):
    print(f"{i}. {exp}")
experiment_choice  = int(input("Digite o número do experimento: ")) - 1
experiment         = experiments[experiment_choice]

print("Escolha um participante:")
for i, pat in enumerate(participantes, 1):
    print(f"{i}. {pat}")
participant_choice = int(input("Digite o número do participante: ")) - 1
participant        = participantes[participant_choice]

session            = input("Digite o número da sessão: ")

print('\nVocê escolheu as seguintes configurações:')
print('EXPERIMENTO: ', experiment)
print('PARTICIPANTE:', participant)
print('SESSÃO:      ', session)

# Caminhos dos ambientes de experimento
path_riemann       = project_path + '/mi processing v4/'
env_monitoring     = path_riemann + '0_signal_monitoring.xml'
env_free           = path_riemann + '1a_Acquisition_Free.xml'
env_screen_2class  = path_riemann + '1b_Acquisition_2_classes.xml'
env_screen_4class  = path_riemann + '1c_Acquisition_4_classes.xml'
env_online_riemann = path_riemann + '3_Online.xml'

# Definir a sequência de blocos do experimento
blocks = [
    ('monitoring',             env_monitoring,   0),
    ('baseline_open_eyes',     env_free,1),
    ('baseline_closed_eyes'   ,env_free,1),
    ('screening_2_classes_mi', env_screen_2class,1),
    # ('online_2_classes_mi'   , env_online_riemann,1),
    ('screening_4_classes_me', env_screen_4class,1),
    # ('online_4_classes_me',    env_online_riemann,1),
    ('screening_4_classes_mi', env_screen_4class,1),
    # ('online_4_classes_mi',    env_online_riemann,1),
    # Adicione mais blocos conforme necessário
]

# Executar cada bloco do experimento
for block_name, block_path, save_entry in blocks:
    run_number = 1
    repeat     = 'y'
    while repeat.lower() == 'y':
        
        if save_entry:
            filename     = f"{experiment}_{block_name}_{participant}_Sess{session}_Run{run_number}.csv"
            filename_csv = data_path + participant + f"/{filename}"
            
            # Dados do logbook
            current_data = {
                'Experiment':  experiment,
                'Participant': participant,
                'Session':     session,
                'Run':         run_number,
                'Type_session':block_name,
                'Filename':    filename,
                'Timestamp':   datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'Computer':    computer_name # Substitua por um identificador real do computador, se necessário
            }
            
            # Construir e executar o comando
            command = f'"{openvibe_designer_path}" --play "{block_path}" --define Filename {filename_csv}'
            subprocess.Popen(command, shell=True)
        else:
            # Construir e executar o comando
            command = f'"{openvibe_designer_path}" --play "{block_path}"'
            subprocess.Popen(command, shell=True)
        
        repeat = input("Deseja realizar novamente? (y/n): ")
        if repeat.lower() == 'y':
            run_number += 1
