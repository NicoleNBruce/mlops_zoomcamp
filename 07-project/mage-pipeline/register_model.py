if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter

import wandb
import pickle
import os

@data_exporter
def export_data(model, *args, **kwargs):
    """
    Exports data to some source.

    Args:
        model: The output from the upstream parent block
    
    """
    wandb.login()

    wandb.init(
    project="07-project",  
    name="experiment_2",          
    )

    model_filename = 'random_forest_model.pkl' 
    local_filepath = "./"
    full_path = os.path.join(local_filepath, model_filename)
    with open(full_path, 'wb') as file: 
        pickle.dump(model, file)

    # logging the model to the W&B run
    wandb.log_model(path=full_path, name="RFC")

    wandb.finish()



