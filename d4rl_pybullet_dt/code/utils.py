import os
import torch
import numpy as np
import cv2
import glob

import gym

"""
============================================
                USER INTERFACE                         
============================================
"""

def clear_console():
    # Limpiar la consola
    os.system('cls' if os.name == 'nt' else 'clear')

def print_logo():
    with open(os.path.join(os.getcwd(),'code','logo.txt'), 'r') as file:
        logo = file.read()
    print(logo)

def mostrar_menu_principal():
    print("--------------------- MAIN MENU ---------------------")
    print("1. Train a Decision Transformer from scratch")
    print("2. Load a Pretrained Decision Transformer")
    print("3. Test a Decision Transformer in Gym-V0 Environments")

def navegar_menu_principal():
    while True:
        clear_console()
        print_logo()
        mostrar_menu_principal()
        opcion = input("\nPor favor, seleccione una opción: ")
        # Convertir la entrada del usuario a un número entero
        try:
            opcion = int(opcion)
        except ValueError:
            print("Por favor, introduzca un número válido.")
            continue
        if opcion == 1:
            print("Has seleccionado la opción 1: Entrenar un modelo desde cero")
            # Aquí podrías llamar a una función para entrenar un modelo desde cero
        elif opcion == 2:
            print("Has seleccionado la opción 2: Cargar un modelo preentrenado")
            config = load_config_from_checkpoint()
            #if config:
            #    print(config)
        elif opcion == 3:
            print("Has seleccionado la opción 3: Testear un modelo")
            # Aquí podrías llamar a una función para testear un modelo
        else:
            print("Opción inválida. Por favor, seleccione una opción válida.")


#/////////////////////////////////////////////////////////////////////////



def get_episodes(terminals):
    terminals = terminals.astype('int32')
    #Las posiciones donde estan los Terminal=1
    if terminals[-1] == 0 : 
        terminals[-1] = 1  
    terminal_pos = np.where(terminals==1)[0]
    return terminal_pos.tolist(), len(terminal_pos)

def get_rtgs(t_positions, rewards):
    # Initialize the starting index of the sub-list in B
    start_idx = 0
    rtgs = []

    
    for t in t_positions:
        end_idx = t + 1
        sub_list = rewards[start_idx:end_idx]
        #print(sub_list)
        for i in range(0, len(sub_list)):
            rtgs.append(sum(sub_list[i+1:]))
        start_idx = end_idx
    return rtgs

def optimized_get_rtgs(t_positions, rewards):

    rewards = np.array(rewards, dtype=np.float64)
    t_positions = np.array(t_positions)

    cumsum_rewards = np.cumsum(rewards)
    
    # Initialize an array to hold the RTGs
    rtgs = np.array([], dtype=int)
    
    # Keep track of the start index of the sub-list in rewards
    start_idx = 0
    for end_idx in t_positions:
        
        segment_rtgs = cumsum_rewards[end_idx] - cumsum_rewards[start_idx:end_idx]
        segment_rtgs = np.append(segment_rtgs, 0)
        rtgs = np.concatenate((rtgs, segment_rtgs))
    
        start_idx = end_idx+1
    return rtgs.tolist()

def list_episodes(terminals_idxs):
    episode_ends = np.array(terminals_idxs)
    episode_starts=np.roll(episode_ends, shift=1) + 1
    episode_starts[0] = 0
    return list(zip(episode_starts, episode_ends +1))

def get_timesteps(episodes, size):
    # Initialize the array of timesteps
    arrayTimesteps = np.zeros(size, dtype=int)

    # List to hold the total steps per episode
    steps_per_episode = []

    # Generate timesteps for each episode
    for start, end in episodes:
        episode_length = end - start
        steps_per_episode.append(episode_length)
        arrayTimesteps[start:end] = np.arange(episode_length)
    return arrayTimesteps, steps_per_episode

def normalize_array(array):
    mean = np.mean(array, axis=0)
    std = np.std(array, axis=0)
    norm_array = (array - mean) / (std+1e-6)
    return norm_array, mean, std

def get_data_set(obs, actions, rewards, terminals, episodes):
    d_obs = [obs[start:end] for start, end in episodes]
    d_act = [actions[start:end] for start, end in episodes]
    d_rew = [rewards[start:end] for start, end in episodes]
    d_ter = [terminals[start:end] for start, end in episodes]

    r_obs = np.concatenate(d_obs, axis=0)
    r_act = np.concatenate(d_act, axis=0)
    r_rew = np.concatenate(d_rew, axis=0)
    r_ter = np.concatenate(d_ter, axis=0)

    return r_obs, r_act, r_rew, r_ter

#O.A 26.02.2024
def discounted_returns(rewards, discount_rate=0.99):
    discounted = np.zeros_like(rewards, dtype=np.float32)
    running_add = 0
    for t in reversed(range(len(rewards))):
        running_add = running_add * discount_rate + rewards[t]
        discounted[t] = running_add
    return discounted

def get_video_filename(video_dir): #No necesaria
  glob_mp4 = os.path.join(dir, "*.mp4")
  mp4list = glob.glob(glob_mp4)
  assert len(mp4list) > 0, "couldnt find video files"
  return mp4list[-1]


def generate_video_opencv(video_frames, video_name):    

    # Video Dimensions according input frames size + Frames per second
    height, width, _ = video_frames[0].shape #240x320
    fps = 40

    videos_directory  = os.path.join(os.getcwd(),"videos")

    #path_save_video = os.path.join(os.getcwd(), 'videos', video_name+'.mp4')
    os.makedirs(videos_directory, exist_ok=True)  
    print("Path where the videos will be stored:", videos_directory)
   
    #Check for existing files (Avoid Overwritting)
    
    count=1
    
    video_file = os.path.join(videos_directory, video_name+'.mp4')

    while os.path.exists(video_file):
        video_file = os.path.join(videos_directory, video_name+f'_{count}.mp4')
        count += 1

    # Initialize video Writer
    fourcc = cv2.VideoWriter_fourcc(*'avc1') # H.264 (AVC) with container MP4 for video compression
    video_writer = cv2.VideoWriter(video_file, fourcc, fps, (width, height))
    
    # Add frames to the video
    for frame in video_frames:
        # Convert frames to BGR format
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # Write the frame in the video
        video_writer.write(frame_bgr)

    # Release video writer
    video_writer.release()
    print("Video succesfully stored at: ", video_file)

    return video_file
    
def generate_video_opencv_v2(video_frames, video_name):

    #List with sharpenning kernels
    sharpening_kernel = [
    np.array([[-1, -1, -1],
                [-1,  9, -1],
                [-1, -1, -1]]),  # (Focus) Unsharp mask kernel

    np.array([[0, -1, 0],
                [-1, 5, -1],
                [0, -1, 0]]),    # Edge enhancement kernel

    np.array([[0, -1, 0],
                [-1, 9, -1],
                [0, -1, 0]])     # Detail enhancement kernel
    ]

    # Video Dimensions according input frames size + Frames per second
    height, width, _ = video_frames[0].shape #240x320
    fps = 40

    videos_directory  = os.path.join(os.getcwd(),"videos")

    #path_save_video = os.path.join(os.getcwd(), 'videos', video_name+'.mp4')
    os.makedirs(videos_directory, exist_ok=True)  
    print("Path where the videos will be stored:", videos_directory)
   
    #Check for existing files (Avoid Overwritting)
    
    count=1
    
    video_file = os.path.join(videos_directory, video_name+'_hd'+'.mp4')

    while os.path.exists(video_file):
        video_file = os.path.join(videos_directory, video_name+f'_hd_{count}.mp4')
        count += 1

    # Configurar el escritor de video
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    video_writer = cv2.VideoWriter(video_file, fourcc, fps, (width, height))  # Resolución de 240x320 y 40 fps
    
    # Bucle para procesar cada frame del video
    for frame in video_frames:
        # Escalar el frame a una resolución más alta 
        #frame_resized = cv2.resize(frame, (720, 480)) #====>> Can't be scalated (Default 240x320)
        
        # Aplicar afilado para resaltar bordes y detalles
        frame_afilado = cv2.filter2D(frame, -1, sharpening_kernel[1]) # Edge enhancement kernel as preference
        
        # Aplicar suavizado Gaussiano al frame para mejorar la calidad visual
        frame_suavizado = cv2.GaussianBlur(frame_afilado, (5, 5), 0)

        # Ajustar el contraste y el brillo para mejorar la claridad
        #frame_contrastado = cv2.convertScaleAbs(frame_afilado, alpha=1.2, beta=10)
        
        # Convertir el frame de RGB a BGR (formato de color utilizado por OpenCV)
        frame_bgr = cv2.cvtColor(frame_suavizado, cv2.COLOR_RGB2BGR)
        
        # Escribir el frame en el video
        video_writer.write(frame_bgr)
    
    # Release video writer
    video_writer.release()
    print("Video succesfully stored at: ", video_file)

    return video_file

def load_config_from_checkpoint():
    # Obtener la lista de archivos en el directorio actual
    files = os.listdir(os.path.join(os.getcwd(),'checkpoints'))
    
    # Filtrar solo los archivos .pt
    pt_files = [f for f in files if f.endswith('.pt')]
    
    if not pt_files:
        print("No hay archivos .pt en el directorio actual.")
        return None
    
    print("Archivos .pt disponibles:")
    for i, f in enumerate(pt_files, start=1):
        print(f"{i}. {f}")
    
    # Pedir al usuario que seleccione un archivo por su índice
    while True:
        try:
            selection = int(input("Seleccione el número correspondiente al archivo .pt que desea cargar: "))
            if 1 <= selection <= len(pt_files):
                break
            else:
                print("Número fuera de rango. Por favor, seleccione un número válido.")
        except ValueError:
            print("Entrada inválida. Por favor, introduzca un número válido.")
    
    # Obtener el nombre del archivo seleccionado
    file_name = pt_files[selection - 1]
    
    # Cargar el archivo .pt seleccionado
    checkpoint = torch.load(os.path.join(os.getcwd(),'checkpoints',file_name), map_location=torch.device('cpu'))
    
    # Obtener la configuración del modelo
    config = checkpoint['config']
    model_state_dict = checkpoint['model_state_dict']
    optimizer_state_dict = checkpoint['optimizer_state_dict']
    env_name = checkpoint['env_name']

    clear_console()
    print_logo()
    print("Models config:",config,"\nEnvironment name: ",env_name)
    user_input = input("Por favor, inserte un dato: ")
    print("Ha insertado:", user_input)

    return config

