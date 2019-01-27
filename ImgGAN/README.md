# Conditional Image GAN for Imputing Imagery Attributes

Implementations of generating images from Multi-modal Embeddings.

## Usage

Use face detectors to crop faces from the augmented YAGO dataset

Convert images to tfrecords format:
options are hard-coded

```
$ python convert.py
```

Train:   
Hyperparameters are contained in the specifications for each model under models folder, 
which are not exposed to execution arguments.


```
$ python --model ECBEGAN -D yago_facecrop --ckpt_step 150 

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         ECBEGAN
  --dataset DATASET, -D DATASET
                        yago_facecrop
  --ckpt_step CKPT_STEP
                        # of steps for saving checkpoint (default: 5000)
```

Monitor through TensorBoard:

```
$ tensorboard --logdir=summary/dataset/name
```

Evaluate (generate samples on **TEST SET**):

```
$ python --model ECBEGAN --dataset yago_facecrop --sample 300 --rep 5

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         ECBEGAN
  --dataset DATASET, -D DATASET
                        yago_facecrop
  --sample_size SAMPLE_SIZE, -N SAMPLE_SIZE
                        # of samples. This is the number of entities selected
  --rep REPITITIONS     # of random seed sampled for each entity
```


### Requirements

- python 3.5
- tensorflow >= 1.4 (verified on 1.4)
- tqdm
- (optional) pynvml - for automatic gpu selection


