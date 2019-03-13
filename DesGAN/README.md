# Conditional Text GAN for Imputing Description Attributes

Implementations of generating descriptions from Multi-modal Embeddings.

# Usage

step_0 : The description of entities are in the ./data. Put the learned embedding of the entities in the same folder, and make sure the embeddings align with their corresponding descriptions. <br /> 

step_1 (optional): Install the KenLM (instruction for the installing the KenLM can be find in [Zhao et al., 2017](https://github.com/jakezhaojb/ARAE)). <br />

step_2 : train the model using: <br />
 > python train.py --data_path ./data --save_path ./output



# Dependency

The description generation is based on ARAE-GAN model from [Zhao et al., 2017](https://github.com/jakezhaojb/ARAE)
