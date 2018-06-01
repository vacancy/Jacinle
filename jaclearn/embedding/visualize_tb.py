#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : visualize_tb.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 03/22/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import os.path as osp

import numpy as np
import tensorflow as tf

from tensorflow.contrib.tensorboard.plugins import projector

__all__ = ['visualize_word_embedding_tb']


def visualize_word_embedding_tb(emb, log_dir):
    # https://stackoverflow.com/questions/41258391/tensorboard-embedding-example#answer-42676076

    if isinstance(emb, tuple):  # embedding, word2idx
        words = sorted(emb[1].keys(), key=lambda x: emb[1][x])
        embedding = np.array(emb[0])
    else:
        words = emb.keys()
        embedding = np.stack([emb[key] for key in words])

    # setup a TensorFlow session
    tf.reset_default_graph()
    embedding_var = tf.Variable(embedding, name='embedding')

    with open(osp.join(log_dir, 'metadata.tsv'), 'w') as f:
        for w in words:
            f.write(w + '\n')

    # create a TensorFlow summary writer
    summary_writer = tf.summary.FileWriter(log_dir, tf.get_default_graph())
    config = projector.ProjectorConfig()
    embedding_conf = config.embeddings.add()
    embedding_conf.tensor_name = embedding_var.name
    embedding_conf.metadata_path = 'metadata.tsv'
    projector.visualize_embeddings(summary_writer, config)

    # save the model
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(embedding_var.initializer)
        saver.save(sess, osp.join(log_dir, "model.ckpt"))
