from __future__ import division
from __future__ import print_function

import argparse
import sys

import numpy as np
import tensorflow as tf

from dataset_utils import DatasetLoader
from retrieval_model import setup_train_model

FLAGS = None

def main(_):
    # Load data.
    data_loader = DatasetLoader(FLAGS.image_feat_path, FLAGS.sent_feat_path)
    num_ims, im_feat_dim = data_loader.im_feat_shape
    num_sents, sent_feat_dim = data_loader.sent_feat_shape
    steps_per_epoch = num_sents // FLAGS.batch_size
    num_steps = steps_per_epoch * FLAGS.max_num_epoch

    # Setup placeholders for input variables.
    im_feat_plh = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, im_feat_dim])
    sent_feat_plh = tf.placeholder(tf.float32, shape=[FLAGS.batch_size * FLAGS.sample_size, sent_feat_dim])
    label_plh = tf.placeholder(tf.bool, shape=[FLAGS.batch_size * FLAGS.sample_size, FLAGS.batch_size])
    train_phase_plh = tf.placeholder(tf.bool)

    # Setup training operation.
    loss = setup_train_model(im_feat_plh, sent_feat_plh, train_phase_plh, label_plh, FLAGS)

    # Setup optimizer.
    global_step = tf.Variable(0, trainable=False)
    init_learning_rate = 0.0001
    learning_rate = tf.train.exponential_decay(init_learning_rate, global_step,
                                               steps_per_epoch, 0.794, staircase=True)
    optim = tf.train.AdamOptimizer(init_learning_rate)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = optim.minimize(loss, global_step=global_step)

    # Setup model saver.
    saver = tf.train.Saver(save_relative_paths=True)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        if FLAGS.restore_path:
            print('restoring checkpoint', restore_path)
            saver.restore(sess, restore_path.replace('.meta', ''))
            print('done')

        for i in range(num_steps):
            if i % steps_per_epoch == 0:
                # shuffle the indices.
                data_loader.shuffle_inds()
            im_feats, sent_feats, labels = data_loader.get_batch(
                    i % steps_per_epoch, FLAGS.batch_size, FLAGS.sample_size)
            feed_dict = {
                    im_feat_plh : im_feats,
                    sent_feat_plh : sent_feats,
                    label_plh : labels,
                    train_phase_plh : True,
            }
            [_, loss_val] = sess.run([train_step, loss], feed_dict = feed_dict)
            if i % 50 == 0:
                print('Epoch: %d Step: %d Loss: %f' % (i // steps_per_epoch, i, loss_val))
            if i % steps_per_epoch == 0 and i > 0:
                print('Saving checkpoint at step %d' % i)
                saver.save(sess, FLAGS.save_dir, global_step = global_step)


if __name__ == '__main__':
    np.random.seed(0)
    tf.set_random_seed(0)

    parser = argparse.ArgumentParser()
    # Dataset and checkpoints.
    parser.add_argument('--image_feat_path', type=str, help='Path to the image feature mat file.')
    parser.add_argument('--sent_feat_path', type=str, help='Path to the sentence feature mat file.')
    parser.add_argument('--save_dir', type=str, help='Directory for saving checkpoints.')
    parser.add_argument('--restore_path', type=str, help='Path to the restoring checkpoint MetaGraph file.')
    # Training parameters.
    parser.add_argument('--batch_size', type=int, default=500, help='Batch size for training.')
    parser.add_argument('--sample_size', type=int, default=2, help='Number of positive pair to sample.')
    parser.add_argument('--max_num_epoch', type=int, default=20, help='Max number of epochs to train.')
    parser.add_argument('--num_neg_sample', type=int, default=10, help='Number of negative example to sample.')
    parser.add_argument('--margin', type=float, default=0.05, help='Margin.')
    parser.add_argument('--im_loss_factor', type=float, default=1.5,
                        help='Factor multiplied with image loss. Set to 0 for single direction.')
    parser.add_argument('--sent_only_loss_factor', type=float, default=0.05,
                        help='Factor multiplied with sent only loss. Set to 0 for no neighbor constraint.')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
