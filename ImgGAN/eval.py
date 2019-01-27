#coding: utf-8
import tensorflow as tf
import numpy as np
import utils, cv2
import config, pickle
import os, glob
import scipy.misc
import random
from argparse import ArgumentParser
slim = tf.contrib.slim


def build_parser():
    parser = ArgumentParser()
    models_str = ' / '.join(config.model_zoo)
    parser.add_argument('--model', help=models_str, required=True) 
    parser.add_argument('--name', help='default: name=model')
    parser.add_argument('--dataset', '-D', required=True)
    parser.add_argument('--sample', '-N', help='# of samples. It should be a square number. (default: 16)',
        default=16, type=int)
    parser.add_argument('--rep', default=16)

    return parser


def sample_z(shape):
    return np.random.normal(size=shape)


def get_all_checkpoints(ckpt_dir, step=None, force=False):
    '''
    When the learning is interrupted and resumed, all checkpoints can not be fetched with get_checkpoint_state 
    (The checkpoint state is rewritten from the point of resume). 
    This function fetch all checkpoints forcely when arguments force=True.
    '''

    if force:
        ckpts = os.listdir(ckpt_dir) # get all fns
        ckpts = map(lambda p: os.path.splitext(p)[0], ckpts) # del ext
        ckpts = set(ckpts) # unique
        ckpts = filter(lambda x: x.split('-')[-1].isdigit(), ckpts) # filter non-ckpt
        ckpts = sorted(ckpts, key=lambda x: int(x.split('-')[-1])) # sort
        if step is not None:
            ckpts = filter(lambda x: x.split('-')[-1] == str(step), ckpts)
        ckpts = list(map(lambda x: os.path.join(ckpt_dir, x), ckpts)) # fn => path
    else:
        ckpts = tf.train.get_checkpoint_state(ckpt_dir).all_model_checkpoint_paths

    return ckpts


def load_meta():
    token2id = np.load("../assets/yago-weights/token2id.npy").item(0)
    entity_emb = np.load("../assets/yago-weights/entity_embedding.npy")
    entity_emb /= np.std(entity_emb)
    return token2id, entity_emb


class CSampler:
    def __init__(self):
        self.token2id = np.load("../assets/yago-weights/token2id.npy").item(0)
        self.entity_emb = np.load("../assets/yago-weights/entity_embedding.npy")
        self.entity_emb /= np.std(self.entity_emb)
        self.tokens = list(self.token2id.keys())
        with open("../assets/yago-test-imgid.pkl", "rb") as f:
            self.testids = pickle.load(f)

    def sample_c(self, rep):
        c_l = []
        ind_l = []
        for ind in self.testids:
            for rep_ind in range(rep):
                c_l.append(self.entity_emb[ind, :])
                ind_l.append((ind, rep_ind))

        return np.array(c_l), ind_l


class Estimator:
    def __init__(self):
        self.token2id = np.load("../assets/yago-weights/token2id.npy").item(0)
        self.id2token = np.load("../assets/yago-weights/id2token.npy").item(0)
        self.entity_emb = np.load("../assets/yago-weights/entity_embedding.npy")
        self.entity_emb /= np.std(self.entity_emb)
        with open("../assets/yago-test-imgid.pkl", "rb") as f:
            self.testids = pickle.load(f)

    def sample_c(self, rep):
        for ind in self.testids:
            yield np.array([self.entity_emb[ind, :]] * rep, dtype=np.float32), ind

    def sample_c_ind(self, rep):
        for ind in self.testids:
            yield np.array([ind] * rep, dtype=np.int64), ind

    def dump_step(self, sess, model, step, dir, rep=4, merge_lim=20, modelname="ECBEGAN"):
        tf.gfile.MakeDirs(os.path.join(dir, str(step)))
        print(modelname)

        merge_list = []
        generator = self.sample_c_ind(rep) if modelname == "ECBEGAN" else self.sample_c(rep)
        for c, ind in generator:
            z = sample_z([rep, model.z_dim])

            if modelname == "ECBEGAN":
                gen_batch = ((sess.run(
                    model.fake_sample, {model.z: z, model.imgid: c}) + 1.0) / 2.0 * 255.0).astype(np.int32)
            else:
                gen_batch = ((sess.run(
                    model.fake_sample, {model.z: z, model.c: c}) + 1.0) / 2.0 * 255.0).astype(np.int32)

            if len(merge_list) < merge_lim:
                merge_list.append(gen_batch)

            for rep_ind in range(rep):
                img = gen_batch[rep_ind, ...]
                cv2.imwrite(os.path.join(dir, str(step), "{}.{}.png".format(ind, rep_ind)), img[..., ::-1])
        merge_blob = np.concatenate(merge_list, 0)
        merged_img = utils.merge(merge_blob, size=[merge_lim, rep])
        cv2.imwrite(os.path.join(dir, "{}.png".format(step)), merged_img[..., ::-1])


def eval(model, name, dataset, sample_shape=[4,4], load_all_ckpt=True):
    if name == None:
        name = model.name
    dir_name = os.path.join('eval', dataset, name)
    if tf.gfile.Exists(dir_name):
        tf.gfile.DeleteRecursively(dir_name)
    tf.gfile.MakeDirs(dir_name)

    sampler = CSampler()

    restorer = tf.train.Saver(slim.get_model_variables())

    config = tf.ConfigProto()
    best_gpu = utils.get_best_gpu()
    config.gpu_options.visible_device_list = str(best_gpu)
    config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_2
    with tf.Session(config=config) as sess:
        ckpt_path = os.path.join('checkpoints', dataset, name)
        ckpts = get_all_checkpoints(ckpt_path, step=None, force=load_all_ckpt)
        size = sample_shape[0] * sample_shape[1]

        z_ = sample_z([size, model.z_dim])
        c, toks = sampler.sample_c(sample_shape[0], sample_shape[1])

        for v in ckpts:
            print("Evaluating {} ...".format(v))
            restorer.restore(sess, v)
            global_step = int(v.split('/')[-1].split('-')[-1])
            
            fake_samples = sess.run(model.fake_sample, {model.z: z_, model.c: c})

            # inverse transform: [-1, 1] => [0, 1]
            fake_samples = (fake_samples + 1.) / 2.
            merged_samples = utils.merge(fake_samples, size=sample_shape)
            fn = "{:0>6d}.png".format(global_step)
            scipy.misc.imsave(os.path.join(dir_name, fn), merged_samples)


def eval_individual(model, name, dataset, num=100, rep=4, step=35000):
    if name == None:
        name = model.name
    dir_name = os.path.join('eval', dataset, name)
    if tf.gfile.Exists(dir_name):
        tf.gfile.DeleteRecursively(dir_name)
    tf.gfile.MakeDirs(dir_name)

    sampler = CSampler()

    restorer = tf.train.Saver(slim.get_model_variables())

    config = tf.ConfigProto()
    best_gpu = utils.get_best_gpu()
    config.gpu_options.visible_device_list = str(best_gpu)
    config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_2
    with tf.Session(config=config) as sess:
        ckpt_path = os.path.join('checkpoints', dataset, name)
        ckpt = get_all_checkpoints(ckpt_path, step=30000, force=True)[0]

        print("Evaluating {} ...".format(ckpt))
        restorer.restore(sess, ckpt)
        global_step = int(ckpt.split('/')[-1].split('-')[-1])
        size = num * rep

        z_ = sample_z([size, model.z_dim])
        c, inds = sampler.sample_c(rep)

        for ind in range(size):
            fake_sample = sess.run(model.fake_sample, {model.z: z_[np.newaxis, ind, :], model.c: c[np.newaxis, ind, :]})
            img = (fake_sample + 1.0) / 2.0
            tokind, rep_ind = inds[ind]
            scipy.misc.imsave(os.path.join(dir_name, "{}.{}.png".format(tokind, rep_ind)), img[0, ...])


def eval_dump(model, name, dataset, rep=4, step=97200):
    dir_name = os.path.join("eval", dataset, name)
    if tf.gfile.Exists(dir_name):
        tf.gfile.DeleteRecursively(dir_name)
    tf.gfile.MakeDirs(dir_name)

    est = Estimator()
    restorer = tf.train.Saver(slim.get_model_variables())

    config = tf.ConfigProto()
    config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_2
    with tf.Session(config=config) as sess:
        ckpt_path = os.path.join('checkpoints', dataset, name)
        ckpt = get_all_checkpoints(ckpt_path, step=step, force=True)[0]

        print("Evaluating {} ...".format(ckpt))
        restorer.restore(sess, ckpt)

        est.dump_step(sess, model, step, dir_name, rep=rep, merge_lim=20)

'''
You can create a gif movie through imagemagick on the commandline:
$ convert -delay 20 eval/* movie.gif
'''
# def to_gif(dir_name='eval'):
#     images = []
#     for path in glob.glob(os.path.join(dir_name, '*.png')):
#         im = scipy.misc.imread(path)
#         images.append(im)

#     # make_gif(images, dir_name + '/movie.gif', duration=10, true_image=True)
#     imageio.mimsave('movie.gif', images, duration=0.2)


if __name__ == "__main__":
    parser = build_parser()
    FLAGS = parser.parse_args()
    FLAGS.model = FLAGS.model.upper()
    FLAGS.dataset = FLAGS.dataset.lower()
    if FLAGS.name is None:
        FLAGS.name = FLAGS.model.lower()
    config.pprint_args(FLAGS)

    N = int(FLAGS.sample)
    rep = int(FLAGS.rep)

    # training=False => build generator only
    model = config.get_model(FLAGS.model, FLAGS.name.upper(), training=False)
    eval_dump(model, name=FLAGS.name.upper(), dataset=FLAGS.dataset, rep=5)
