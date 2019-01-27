# coding: utf-8

import tensorflow as tf
import numpy as np
import scipy.misc
import os, cv2
import glob


def _bytes_features(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _int64_features(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _floats_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))



def convert_yago(source_dir, target_dir, crop_size, out_size, exts=[''], num_shards=128, tfrecords_prefix=''):
    if not tf.gfile.Exists(source_dir):
        print('source_dir does not exists')
        return
    
    if tfrecords_prefix and not tfrecords_prefix.endswith('-'):
        tfrecords_prefix += '-'

    if tf.gfile.Exists(target_dir):
        print("{} is Already exists".format(target_dir))
        return
    else:
        tf.gfile.MakeDirs(target_dir)

    # get meta-data
    path_list = []
    for ext in exts:
        pattern = '*.' + ext if ext != '' else '*'
        path = os.path.join(source_dir, pattern)
        path_list.extend(glob.glob(path))

    # read embeddings
    token2id = np.load("../assets/yago-weights/token2id.npy").item(0)
    entity_emb = np.load("../assets/yago-weights/entity_embedding.npy")
    entity_emb /= np.std(entity_emb)

    # shuffle path_list
    np.random.shuffle(path_list)
    num_files = len(path_list)
    num_per_shard = num_files // num_shards # Last shard will have more files

    print('# of files: {}'.format(num_files))
    print('# of shards: {}'.format(num_shards))
    print('# files per shards: {}'.format(num_per_shard))

    # convert to tfrecords
    shard_idx = 0
    writer = None
    for i, path in enumerate(path_list):
        if i % num_per_shard == 0 and shard_idx < num_shards:
            shard_idx += 1
            tfrecord_fn = '{}{:0>4d}-of-{:0>4d}.tfrecord'.format(tfrecords_prefix, shard_idx, num_shards)
            tfrecord_path = os.path.join(target_dir, tfrecord_fn)
            print("Writing {} ...".format(tfrecord_path))
            if shard_idx > 1:
                writer.close()
            writer = tf.python_io.TFRecordWriter(tfrecord_path)

        # mode='RGB' read even grayscale image as RGB shape
        im = cv2.imread(path)[..., ::-1]
        if im.shape != (64, 64, 3):
            raise ValueError("Incompatible shape {}".format(str(im.shape)))

        # get embedding
        name = os.path.splitext(os.path.basename(path))[0].lower()
        emb = entity_emb[token2id[name], :]

        example = tf.train.Example(features=tf.train.Features(feature={
            # "shape": _int64_features(im.shape),
            "image": _bytes_features([im.tostring()]),
            "emb": _floats_feature(emb),
            "imgid": _int64_features([token2id[name]])
        }))
        writer.write(example.SerializeToString())

    writer.close()


def convert(source_dir, target_dir, crop_size, out_size, exts=[''], num_shards=128, tfrecords_prefix=''):
    if not tf.gfile.Exists(source_dir):
        print('source_dir does not exists')
        return

    if tfrecords_prefix and not tfrecords_prefix.endswith('-'):
        tfrecords_prefix += '-'

    if tf.gfile.Exists(target_dir):
        print("{} is Already exists".format(target_dir))
        return
    else:
        tf.gfile.MakeDirs(target_dir)

    # get meta-data
    path_list = []
    for ext in exts:
        pattern = '*.' + ext if ext != '' else '*'
        path = os.path.join(source_dir, pattern)
        path_list.extend(glob.glob(path))

    # shuffle path_list
    np.random.shuffle(path_list)
    num_files = len(path_list)
    num_per_shard = num_files // num_shards  # Last shard will have more files

    print('# of files: {}'.format(num_files))
    print('# of shards: {}'.format(num_shards))
    print('# files per shards: {}'.format(num_per_shard))

    # convert to tfrecords
    shard_idx = 0
    writer = None
    for i, path in enumerate(path_list):
        if i % num_per_shard == 0 and shard_idx < num_shards:
            shard_idx += 1
            tfrecord_fn = '{}{:0>4d}-of-{:0>4d}.tfrecord'.format(tfrecords_prefix, shard_idx, num_shards)
            tfrecord_path = os.path.join(target_dir, tfrecord_fn)
            print("Writing {} ...".format(tfrecord_path))
            if shard_idx > 1:
                writer.close()
            writer = tf.python_io.TFRecordWriter(tfrecord_path)

        # mode='RGB' read even grayscale image as RGB shape
        im = scipy.misc.imread(path, mode='RGB')
        # preproc
        try:
            im = center_crop(im, crop_size)
        except Exception as e:
            # print("im_path: {}".format(path))
            # print("im_shape: {}".format(im.shape))
            print("[Exception] {}".format(e))
            continue

        im = scipy.misc.imresize(im, out_size)
        example = tf.train.Example(features=tf.train.Features(feature={
            # "shape": _int64_features(im.shape),
            "image": _bytes_features([im.tostring()])
        }))
        writer.write(example.SerializeToString())

    writer.close()


if __name__ == "__main__":
    # YAGO
    convert_yago('../assets/yago-facecrop', '../assets/yago-facecrop-tfrecord', crop_size=[128, 128], out_size=[128, 128],
        exts=['jpg', 'png'], num_shards=128, tfrecords_prefix='yago')


