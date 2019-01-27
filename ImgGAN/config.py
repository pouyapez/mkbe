from models import *


model_zoo = ['EBGAN', 'BEGAN', 'DRAGAN']

def get_model(mtype, name, training):
    model = None
    if mtype == 'EBGAN':
        model = ebgan.EBGAN
    elif mtype == 'BEGAN':
        model = began.BEGAN
    elif mtype == 'CBEGAN':
        model = cbegan.BEGAN
    elif mtype == 'CBEGANHG':
        model = cbeganhg.BEGAN
    elif mtype == 'ECBEGAN':
        model = ecbegan.BEGAN
    else:
        assert False, mtype + ' is not in the model zoo'

    assert model, mtype + ' is work in progress'

    return model(name=name, training=training)


def get_dataset(dataset_name):
    yago_64 = "../assets/YAGO_imgs/*.jpg"
    yago_tfrecord = "../assets/YAGO_imgs_tfrecord/*.tfrecord"
    yago_facecrop = "../assets/yago-facecrop-tfrecord/*.tfrecord"

    if dataset_name == 'yago':
        path = yago_64
        n_examples = 28950
    elif dataset_name == 'yago_tfrecord':
        path = yago_tfrecord
        n_examples = 28950
    elif dataset_name == "yago_facecrop":
        path = yago_facecrop
        n_examples = 21789
    else:
        raise ValueError('{} is does not supported. dataset must be celeba or lsun.'.format(dataset_name))

    return path, n_examples


def pprint_args(FLAGS):
    print("\nParameters:")
    for attr, value in sorted(vars(FLAGS).items()):
        print("{}={}".format(attr.upper(), value))
    print("")

