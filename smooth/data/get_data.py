import numpy as np
from datasets import load_dataset, logging, DownloadConfig
# from datasets import clear_cache
# clear_cache()
# logging.set_verbosity_info()
logging.set_verbosity_debug()



# name_save = "inverted_pendulum_dataset.npz"
# name_load = "InvertedPendulum-v4"

# name_save = "walker_dataset.npz"
# name_load = "Walker2d-v4"

# name_save = "half_cheetah_dataset.npz"
# name_load = "HalfCheetah-v4"

name_save = "swimmer_dataset.npz"
name_load = "Swimmer-v4"


download_config = DownloadConfig(
    resume_download=True,
    max_retries=5,
    force_download=True,
    delete_extracted=True
)

# Load the dataset
dataset = load_dataset("NathanGavenski/"+name_load, 
                       trust_remote_code=True, 
                        download_config=download_config
                       )
train_dataset = dataset['train']

# Save the dataset

np.savez(name_save, 
         obs=train_dataset['obs'], 
         actions=train_dataset['actions'], 
         episode_starts=train_dataset['episode_starts'])
