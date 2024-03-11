import os
import torch
import numpy as np
import cv2
import glob
import datetime

from scipy.stats import linregress


"""
============================================
              ENVIRONMENTS DICT                         
============================================
"""

env_dict = {

    # Hopper Agent
    #'hopper-bullet-random-v0': None, # ==> Contains NaN values
    'hopper-bullet-medium-v0': None,
    'hopper-bullet-mixed-v0': None,

    # HalfCheetah Agent
    #'halfcheetah-bullet-random-v0': None, ==> Contains NaN values
    'halfcheetah-bullet-medium-v0': None,
    'halfcheetah-bullet-mixed-v0': None,

    # Ant Agent
    #'ant-bullet-random-v0': None, # ==> Contains NaN values
    'ant-bullet-medium-v0': None,
    'ant-bullet-mixed-v0': None,

    # Walker2D Agent
    #'walker2d-bullet-random-v0': None, # ==> Contains NaN values
    'walker2d-bullet-medium-v0': None,
    'walker2d-bullet-mixed-v0': None
}



"""
============================================
                USER INTERFACE                         
============================================
"""

def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_logo():
    #with open(os.path.join(os.getcwd(),'code','logo.txt'), 'r') as file:
    with open(os.path.join(os.getcwd(),'logo.txt'), 'r') as file:
        logo = file.read()
    print(logo)

def display_main_menu():
    print("  --------------------- MAIN MENU ---------------------")
    print("  1. Train a Decision Transformer from scratch")
    print("  2. Overview Pretrained Decision Transformer config")
    print("  3. Test a Decision Transformer in Gym-V0 Environments")
    print("  4. Exit runtime execution")



def show_training_environments():
    print('  ├→ Available environments for training:')
    print("\r")
    for i, env_name in enumerate(env_dict.keys(), start=1):
        print(f"     {i}. {env_name}")
    
    while True:
        try:
            selection = int(input("\n  Select an environment by entering its number: "))
            if 1 <= selection <= len(env_dict):
                selected_env = list(env_dict.keys())[selection - 1]
                return selected_env
            else:
                print("  Invalid selection. Please enter a valid number.")
        except ValueError:
            print("  Invalid input. Please enter a number.")

def show_testing_environments():
    print('  ├→ Available environments for testing:')
    print("\r")
    for i, env_name in enumerate(env_dict.keys(), start=1):
        print(f"     {i}. {env_name}")
    
    while True:
        try:
            selection = int(input("\n  Select an environment by entering its number: "))
            if 1 <= selection <= len(env_dict):
                selected_env = list(env_dict.keys())[selection - 1]
                return selected_env
            else:
                print("  Invalid selection. Please enter a valid number.")
        except ValueError:
            print("  Invalid input. Please enter a number.")

def show_configurations():
    
    config, env_name, _ = load_config_from_checkpoint()
    if config and env_name:
        print("\n  ╔═══════════════════════════════════════╗")
        print("  ║             Models Config             ║")
        print("  ╚═══════════════════════════════════════╝")
        for key, value in config.items():
            print(f"   →  {key}: {value}")

        print("\n  ╔═══════════════════════════════════════╗")
        print("  ║     Agent's Training Environment      ║")
        print("  ╚═══════════════════════════════════════╝")
        print(f"   → Environment name: {env_name}")
        
        input("\n  Press ENTER to return to Main Menu.")
        navigate_main_menu()
    else:
        print("  Model's configuration or environment name not available.")


def navigate_main_menu():
    clear_console()
    print_logo()
    display_main_menu()

    option_selected = False
    trigger_train = False   
    trigger_test = False 
    env_name = None  
    pretrained_file_name = None
    
    
    while not option_selected:

        # Convert input to integer
        try:
            option = input("\n  Please, select one of the options available: ")
            option = int(option)
        except ValueError:
            clear_console()
            print_logo()
            display_main_menu()
            continue
        
        #==> Option 1
        if option == 1:
            clear_console()
            print_logo()

            print("\n  ╔════════════════════════════════════════════════════════╗")
            print("  ║  Selected: Train a Decision Transformer from scratch   ║")
            print("  ╚════════════════════════════════════════════════════════╝")
            env_name = show_training_environments()
            trigger_train = True
            #Jump to main
            option_selected = True
            
        
        #==> Option 2    
        elif option == 2:
            clear_console()
            print_logo()
            
            print("\n  ╔═════════════════════════════════════════════════════════╗")
            print("  ║  Selected: Overview & Edit Environment's model config   ║")
            print("  ╚═════════════════════════════════════════════════════════╝")

            show_configurations()
            
       
        #==> Option 3
        elif option == 3:
            clear_console()
            print_logo()

            print("\n  ╔═════════════════════════════════════════════════════════════════╗")
            print("  ║  Selected: Test a Decision Transformer in Gym-V0 Environments   ║")
            print("  ╚═════════════════════════════════════════════════════════════════╝")

            # Select pretrained file .pt
            config, env_name, pretrained_file_name = load_config_from_checkpoint()
            # Select environment to test
            env_name = show_testing_environments()
            trigger_test = True
            #Jump to main
            option_selected = True
       
        #==> Option Exit   
        elif option == 4:
            clear_console()
            print_logo()
            print("  Exiting the program. Goodbye! :)")
            #Jump to main
            option_selected = True
            

    return trigger_train, trigger_test, env_name, pretrained_file_name


def execution_stats(start_time):
    end_time = datetime.datetime.now()
    execution_time = end_time - start_time

    print("\n  ╔════════════════════════════════════════╗")
    print("  ║  Execution completed at:", end_time.strftime("| %H:%M h |"),"  ║")
    print("  ║  Total execution time:", execution_time, " ║")
    print("  ╚════════════════════════════════════════╝")


"""
============================================
                FUNCTIONS                         
============================================
"""

def trend_arrow(data, window_size=10):
    if len(data) < window_size:
        return ''

    # Calculated trend via Linear Regression over set of data
    x = range(len(data[-window_size:]))
    y = data[-window_size:]
    slope, _, _, _, _ = linregress(x, y)

    # Data increasing trend
    if slope > 0:
        return '↑'  # 
    # Data decreasing trend
    elif slope < 0:
        return '↓'  
    # Data stability
    else:
        return '~'   


def get_episodes(terminals):
    terminals = terminals.astype('int32')
    #Positions where therminals are located
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

def discounted_returns(rewards, discount_rate=0.99):
    discounted = np.zeros_like(rewards, dtype=np.float32)
    running_add = 0
    for t in reversed(range(len(rewards))):
        running_add = running_add * discount_rate + rewards[t]
        discounted[t] = running_add
    return discounted

def get_video_filename(video_dir): 
  glob_mp4 = os.path.join(dir, "*.mp4")
  mp4list = glob.glob(glob_mp4)
  assert len(mp4list) > 0, "could not find video files"
  return mp4list[-1]


def generate_video_opencv(video_frames, video_name):    

    # Video Dimensions according input frames size + Frames per second
    height, width, _ = video_frames[0].shape #240x320
    fps = 40

    videos_directory  = os.path.join(os.getcwd(),"videos")

    #path_save_video = os.path.join(os.getcwd(), 'videos', video_name+'.mp4')
    os.makedirs(videos_directory, exist_ok=True)  
    print("  Path where the videos will be stored:", videos_directory)
   
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
    print("  Video succesfully stored at: ", video_file)

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
    print("  Path where the videos will be stored:", videos_directory)
   
    #Check for existing files (Avoid Overwritting)
    
    count=1
    
    video_file = os.path.join(videos_directory, video_name+'_hd'+'.mp4')

    while os.path.exists(video_file):
        video_file = os.path.join(videos_directory, video_name+f'_hd_{count}.mp4')
        count += 1

    # Video Writer config
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    video_writer = cv2.VideoWriter(video_file, fourcc, fps, (width, height))  # Resolución de 240x320 y 40 fps
    
    # Loop top process video frames
    for frame in video_frames:
        # Resize every frame to a higher/lower resolution
        #frame_resized = cv2.resize(frame, (720, 480)) #====>> Can't be scalated (Default 240x320) (Dont' use it for this video frames, it crash)
        
        # Apply sharpening to enhance edges and details.
        frame_sharpening = cv2.filter2D(frame, -1, sharpening_kernel[1]) # Edge enhancement kernel as preference
        
        # Apply Gaussian Smooth (blur) to enhance visual quality
        frame_smoothed = cv2.GaussianBlur(frame_sharpening, (5, 5), 0)

        # Adjust brightness and contrast 
        frame_contrasted = cv2.convertScaleAbs(frame_smoothed, alpha=1.2, beta=10)
        
        # Convert the frame from RGB to BGR (color format used by OpenCV)
        frame_bgr = cv2.cvtColor(frame_smoothed, cv2.COLOR_RGB2BGR)
        
        # Write the frame in the video
        video_writer.write(frame_bgr)
    
    # Release video writer
    video_writer.release()
    print("  Video succesfully stored at: ", video_file)

    return video_file

def load_config_from_checkpoint():
    # Obtain a list of files in the current directory
    files = os.listdir(os.path.join(os.getcwd(),'checkpoints'))
    
    # Filter them by .pt files
    pt_files = [f for f in files if f.endswith('.pt')]
    
    if not pt_files:
        print("  No .pt files found in current directory.") 
        return None
    
    print("  ├→ Pretrained model files .pt available:")
    print("\r")
    for i, f in enumerate(pt_files, start=1):
        print(f"     {i}. {f}")
    
    # User selects .pt file
    while True:
        try:
            selection = int(input("\n  Type the number of the .pt file you're interested to load: "))
            if 1 <= selection <= len(pt_files):
                break
            else:
                print("  Número fuera de rango. Por favor, seleccione un número válido.")
        except ValueError:
            print("  Entrada inválida. Por favor, introduzca un número válido.")
    
    # Obtain .pt file name
    file_name = pt_files[selection - 1]
    
    # Load .pt file selected
    checkpoint = torch.load(os.path.join(os.getcwd(),'checkpoints',file_name), map_location=torch.device('cuda'))
    
    # Obtain model's config and env_name 
    config = checkpoint['config']
    model_state_dict = checkpoint['model_state_dict']
    optimizer_state_dict = checkpoint['optimizer_state_dict']
    env_name = checkpoint['env_name']

    return config, env_name, file_name

