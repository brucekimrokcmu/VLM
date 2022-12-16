# Vision and Language Manipulation - CLIPort Reimplementation

This repository is a reimplementation of the cliport vision and language manipulation imitation learning model. All of the pytorch modules can be found in the models directory. The eval_notebooks folder contains jupyter notebooks for evaluating a trained agent and for viewing pick and place affordance maps. The tests folder contains a series of unit tests for checking the input and output sizes of the models. The PickPlaceAgent.py file contains the actual code for a cliport agent, with functions for training, evaluating, and for generating actions. Utils.py contains utils such as classes for keeping track of average statistics and functions for generating tensors to represent the pick and place demos as one-hot images. The train.py file contains code for actually training the agent.

## Setup Instructions
This repo makes use of the dataset, environment, and preprocessing utils from the original cliport implementation. Thus, the first step to installation is to install the original cliport implementation. We ran into out-of-memory issues, so we created a fork of the cliport repo [here](https://github.com/KevinGmelin/cliport) which adds a cache limit. You can instead choose to install the original cliport repo from [here](https://github.com/cliport/cliport).

In the same virtual environment that you installed the original cliport implementation into, you will need to install the CLIP model, following the instructions from [here](https://github.com/openai/CLIP). Our repo has created a wrapper around this CLIP implementation in order to generate the image language embeddings in the semantic stream. 

To generate the dataset for a particular task, follow the instructions from the original cliport repo. 

To start training a cliport agent, first create a directory to store model checkpoints. Then, run train.py, replacing the bold variables below with their true absolute paths.

```sh
mkdir CHECKPOINT_DIR
python train.py --train_data_dir DIR_TO_TRAINING_DATASET --val_data_dir DIR_TO_VAL_DATASET --checkpoint_path CHECKPOINT_DIR
```

You can also enable wandb logging as such:

```sh
mkdir CHECKPOINT_DIR
python train.py --train_data_dir DIR_TO_TRAINING_DATASET --val_data_dir DIR_TO_VAL_DATASET --checkpoint_path CHECKPOINT_DIR \ 
                --wandb --wandb_entity WANDB_ACCOUNT_NAME --wandb_project WANDB_PROJECT_NAME
```

Once an agent has been trained, check out eval_notebooks/eval_pick_place.ipynb for measuring the average success rate of the agent across a large number of evaluation runs. You can also look at eval_notebooks/view_affordances.ipynb to generate and visualize pick and place affordance maps for the agent from a given checkpoint.
