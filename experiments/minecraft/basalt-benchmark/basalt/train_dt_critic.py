import argparse
import glob
from basalt.dt_model import SimpleBinaryClassifier
import torch
from torch.utils.data import random_split

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import os
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm

MAX_DATA_SIZE = 1_000_000


def add_experiment_specific_args(parser):
    parser.add_argument("--embeddings_dir", type=str, required=True, help="Path to the directory that contains the data embeddings")
    parser.add_argument("--output_dir", type=str, required=True, help="Where to store the experiment results")

class EpisodeDataset(Dataset):
    def __init__(self, data_path, expected_embedding_dim=1024, max_files_to_load=None):
        self.expected_embedding_dim = expected_embedding_dim
        self.episodes = self.get_all_npz_files_in_dir(data_path)
        self.max_position = 20
        if max_files_to_load is not None:
            self.episodes = self.episodes[:max_files_to_load]

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):

        episode = self.episodes[idx]
        embeddings, positions = self.load_embedded_trajectories_as_transitions([episode], progress_bar=False, expected_embedding_dim=self.expected_embedding_dim)

        embeddings = torch.from_numpy(embeddings)
        positions = torch.from_numpy(positions).unsqueeze(-1)

        return {'obs': embeddings, 'reward': positions}

    def get_all_npz_files_in_dir(self, dir_path):
        return glob.glob(os.path.join(dir_path, "*.npz"))

    def load_embedded_trajectories_as_transitions(self, npz_file_paths, progress_bar=False, expected_embedding_dim=None):
        all_embeddings_0 = []  
        all_embeddings_1 = []  
        all_labels_binary_0 = []  
        all_labels_binary_1 = []  

        for npz_file_path in tqdm(npz_file_paths, desc="Loading trajectories", disable=not progress_bar, leave=False):
            episode_data = np.load(npz_file_path)
            embeddings = episode_data["embeddings"]
            
            if embeddings.ndim == 1 and expected_embedding_dim is not None:
                embeddings = embeddings.reshape(-1, expected_embedding_dim)

            num_frames = embeddings.shape[0]
            labels_binary = np.zeros(num_frames, dtype=int)
            labels_binary[-self.max_position:] = 1

            
            for i in range(num_frames):
                if labels_binary[i] == 1:
                    all_embeddings_1.append(embeddings[i])
                    all_labels_binary_1.append(labels_binary[i])
                else:
                    all_embeddings_0.append(embeddings[i])
                    all_labels_binary_0.append(labels_binary[i])


        all_embeddings_0 = np.array(all_embeddings_0, dtype=np.float32)
        all_embeddings_1 = np.array(all_embeddings_1, dtype=np.float32)
        all_labels_binary_0 = np.array(all_labels_binary_0, dtype=np.float32)
        all_labels_binary_1 = np.array(all_labels_binary_1, dtype=np.float32)

        num_samples_0 = len(all_embeddings_1)  
        indices_0 = np.random.choice(len(all_embeddings_0), num_samples_0, replace=False)

        balanced_embeddings_0 = all_embeddings_0[indices_0]
        balanced_labels_0 = all_labels_binary_0[indices_0]

        all_embeddings = np.concatenate([balanced_embeddings_0, all_embeddings_1], axis=0)
        all_labels_binary = np.concatenate([balanced_labels_0, all_labels_binary_1], axis=0)

        return all_embeddings, all_labels_binary

def main(args):

    data_path = args.embeddings_dir
    embedding_dim = 1024  
    batch_size = 1
    epochs = 2

    episode_dataset = EpisodeDataset(data_path)
    dataset_size = len(episode_dataset)
    train_size = int(0.9 * dataset_size)
    test_size = dataset_size - train_size
    
    train_dataset, val_dataset = random_split(episode_dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = SimpleBinaryClassifier(embedding_dim)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    def train(epoch):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1} Training")
        for i, data in progress_bar:
            inputs, labels = data['obs'].to(device), data['reward'].to(device)

            inputs = inputs.view(-1, inputs.shape[-1]) 
            labels = labels.view(-1, labels.shape[-1])

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            progress_bar.set_postfix({'loss': running_loss / (i + 1)})
        print(f"Epoch {epoch+1}, Training Loss: {running_loss/len(train_loader)}")

    def validate(epoch):
        model.eval()
        val_loss = 0.0
        total_accuracy = 0.0  
        total_samples = 0  

        progress_bar = tqdm(enumerate(val_loader), total=len(val_loader), desc=f"Epoch {epoch+1} Validation")
        with torch.no_grad():
            for i, data in progress_bar:
                inputs, labels = data['obs'].to(device), data['reward'].to(device)

                
                inputs = inputs.view(-1, inputs.shape[-1])
                labels = labels.view(-1, 1) if labels.dim() == 1 else labels  

                outputs = model(inputs)

                labels = labels.squeeze()  

                # Ahora puedes calcular la pérdida sin error
                loss = criterion(outputs.squeeze(1), labels)
                val_loss += loss.item()

                # Calcula la precisión
                predicted = outputs.round()  
                accuracy = (predicted == labels).float().mean() 
                total_accuracy += accuracy.item() * inputs.size(0)  
                total_samples += inputs.size(0)  

                # Actualiza la barra de progreso con la pérdida de validación hasta el momento
                progress_bar.set_postfix({'val_loss': val_loss / (i + 1)})

        # Calcula la precisión de validación total
        total_accuracy /= total_samples
        print(f"Epoch {epoch+1}, Validation Loss: {val_loss/len(val_loader)}, Validation Accuracy: {total_accuracy}")

    for epoch in range(epochs):
        train(epoch)
        validate(epoch)

    torch.save(model.state_dict(), args.output_dir + '/model_dt_critic.pth')

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train a DT agent on VPT embeddings for the BASALT Benchmark")
    add_experiment_specific_args(parser)
    args = parser.parse_args()
    main(args)