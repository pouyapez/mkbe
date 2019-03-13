# Conditional Text GAN for Imputing Description Attributes

Implementations of generating descriptions from Multi-modal Embeddings.

# Usage

step_0 : The description and the embedding (.npy) of entities are in the ./data.
step_1 (optional): First instal the KenLM (instruction for the installing the KenLM can be find in [Zhao et al., 2017](https://github.com/jakezhaojb/ARAE)).

step_2 : train the model using:
python train.py --data_path ./data --save_path ./output



# Dependency

The description generation is based on ARAE-GAN model from [Zhao et al., 2017](https://github.com/jakezhaojb/ARAE)
