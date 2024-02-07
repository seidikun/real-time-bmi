import subprocess
import os
import csv
from datetime import datetime
# import train_riemann_pca as train_pca

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
            fieldnames = ['Experiment', 'Participant', 'Type_session', 'Session', 'Run', 'Filename', 'Timestamp', 'Computer']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(search_data)
            print("Novo registro adicionado com sucesso.")
    except Exception as e:
        print(f"Erro ao escrever no arquivo {filepath}: {e}")

# Variáveis globais
project_path           = 'C:/Users/Laboratorio/Documents/GitHub/real-time-bmi/mi processing v4'
computer_name          = os.environ['COMPUTERNAME']
openvibe_designer_path = r'C:/Program Files/openvibe-3.5.0-64bit/openvibe-designer.cmd'
data_path              = 'D:/Seidi/'
experiments            = ['RIEMANN_LDA']
participantes          = ['SY100', 'TEST']
logbook_filename       = project_path + '/logbook.csv'

# Configurando sessão atual
print("Escolha um experimento:")
for i, exp in enumerate(experiments, 1):
    print(f"{i}. {exp}")
experiment_choice     = int(input("Digite o número do experimento: ")) - 1
experiment            = experiments[experiment_choice]

print("\nEscolha um participante:")
for i, pat in enumerate(participantes, 1):
    print(f"{i}. {pat}")
participant_choice    = int(input("Digite o número do participante: ")) - 1
participant           = participantes[participant_choice]

session               = input("\nDigite o número da sessão: ")

print('\nVocê escolheu as seguintes configurações:')
print('EXPERIMENTO: ', experiment)
print('PARTICIPANTE:', participant)
print('SESSÃO:      ', session)

# Caminhos dos ambientes de experimento
env_monitoring     = project_path + '/Monitoring.xml'
env_free           = project_path + '/Acquisition_Free.xml'
env_screen_2class  = project_path + '/Acquisition_2_classes_test.xml'
env_screen_4class  = project_path + '/Acquisition_4_classes.xml'
env_online_riemann = project_path + '/Online.xml'

# Definir a sequência de blocos do experimento
blocks = [
    ('monitoring',             env_monitoring,   0),
    # ('baseline_open_eyes',     env_free,1),
    # ('baseline_closed_eyes'   ,env_free,1),
    # ('screening_2_classes_mi', env_screen_2class,1),
    # ('online'   , env_online_riemann,1),
    # ('screening_4_classes_me', env_screen_4class,1),
    # ('online',    env_online_riemann,1),
    # ('screening_4_classes_mi', env_screen_4class,1),
    # ('online',    env_online_riemann,1),
    # Adicione mais blocos conforme necessário
]

def run_without_log(openvibe_designer_path, block_path, filename_csv, current_data):
        # Construir e executar o comando
        command = f'"{openvibe_designer_path}" --play "{block_path}"'
        subprocess.Popen(command, shell=True)
        
        # Espera o usuário teclar Enter após a finalização do bloco
        input("Ao finalizar o bloco, pressione Enter para continuar...")
        
def run_basic(openvibe_designer_path, block_path, filename_csv, current_data, logbook_filename):
        add_record(logbook_filename, current_data)
        # Construir e executar o comando
        command = f'"{openvibe_designer_path}" --play "{block_path}" --define Filename {filename_csv}'
        subprocess.Popen(command, shell=True)
        
        # Espera o usuário teclar Enter após a finalização do bloco
        input("Ao finalizar o bloco, pressione Enter para continuar...")
        
def run_online(openvibe_designer_path, block_path, filename_csv, current_data, logbook_filename, data_train):
        add_record(logbook_filename, current_data)
        # Construir e executar o comando
        command = f'"{openvibe_designer_path}" --play "{block_path}" --define Filename {filename_csv} --define data_train {data_train}'
        subprocess.Popen(command, shell=True)
        
        # Espera o usuário teclar Enter após a finalização do bloco
        input("Ao finalizar o bloco, pressione Enter para continuar...")
        
def select_data_to_train(data_path, experiment, participant, session):
    # Cria o caminho completo para o diretório do participante
    full_path = os.path.join(data_path, participant)
    
    # Lista todos os arquivos no diretório
    all_files = os.listdir(full_path)
    
    # Filtra os arquivos que contêm as substrings necessárias
    filtered_files = [file for file in all_files if experiment in file and 'screening' in file and 'Sess' + session in file]
    
    # Processa os nomes dos arquivos para retornar apenas até 'Sess' + session
    processed_files = set()
    for file in filtered_files:
        # Encontra o índice onde 'Sess' + session termina
        end_index = file.find('Sess' + session) + len('Sess' + session)
        # Extrai a substring do arquivo até esse índice
        processed_file = file[:end_index]
        # Adiciona ao conjunto para garantir unicidade
        processed_files.add(processed_file)
    
    # Converte o conjunto de volta para uma lista para retornar
    return list(processed_files)
    

for block_name, block_path, save_entry in blocks:
    run_number = 1
    repeat     = 'y'
    while repeat.lower() == 'y':
        # Espera o usuário teclar Enter antes de iniciar o bloco
        print('\nVamos fazer um bloco', block_name)
        input("Pressione Enter para abrir o ambiente...")
        
        filename     = f"{experiment}_{block_name}_{participant}_Sess{session}_Run{run_number}"
        filename_csv = data_path + participant + f"/{filename}.csv"
        
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

        if block_name == 'monitoring':
            run_without_log(openvibe_designer_path, block_path, filename_csv, current_data)
        elif block_name == 'baseline_open_eyes':
            run_basic(openvibe_designer_path, block_path, filename_csv, current_data, logbook_filename)
        elif block_name == 'balines_closed_eyes':
            run_basic(openvibe_designer_path, block_path, filename_csv, current_data, logbook_filename)
        elif block_name == 'screening_2_classes_mi':
            run_basic(openvibe_designer_path, block_path, filename_csv, current_data, logbook_filename)
            print('Treinando o modelo...')
            train_pca.process_data(data_path + participant + f"/{filename}", '2classes')
        elif block_name == 'screening_4_classes_mi':
            run_basic(openvibe_designer_path, block_path, filename_csv, current_data, logbook_filename)
            print('Treinando o modelo...')
            train_pca.process_data(data_path + participant + f"/{filename}", '4classes')
        elif block_name == 'screening_4_classes_me':
            run_basic(openvibe_designer_path, block_path, filename_csv, current_data, logbook_filename)
            print('Treinando o modelo...')
            train_pca.process_data(data_path + participant + f"/{filename}", '4classes')
        elif block_name == 'online':
            list_train = select_data_to_train(data_path, experiment, participant, session)
            print('Escolha o dado que você quer usar para o bloco online')
            for i, exp in enumerate(list_train, 1):
                print(f"{i}. {exp}")
            data_choice  = int(input("Digite o número do dado: ")) - 1
            data_train   = list_train[data_choice]
            run_online(openvibe_designer_path, block_path, filename_csv, current_data, logbook_filename, data_train)


        
        # Fechando o OpenViBE Designer após a finalização do bloco
        subprocess.call("taskkill /F /IM openvibe-designer.exe", shell=True)
        
        repeat = input("Deseja realizar novamente? (y/n): ")
        if repeat.lower() == 'y':
            run_number += 1

print("Experimento Finalizado!")