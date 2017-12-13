from __future__ import division
from __future__ import print_function

import argparse
import sys
import time

import numpy as np
import tensorflow as tf

from dataset_utils import DatasetLoader
from retrieval_model import setup_eval_model

FLAGS = None

def eval_once(data_loader, saver, placeholders, recall):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # Restore latest checkpoint or the given MetaGraph.
        if FLAGS.restore_path.endswith('.meta'):
            ckpt_path = FLAGS.restore_path.replace('.meta', '')
        else:
            ckpt_path = tf.train.latest_checkpoint(FLAGS.restore_path)
        print('Restoring checkpoint', ckpt_path)
        saver = tf.train.import_meta_graph(ckpt_path + '.meta')
        saver.restore(sess, ckpt_path)
        print('Done')

        # For testing and validation, there should be only one batch with index 0.
        im_feats, sent_feats, labels = data_loader.get_batch(0, FLAGS.batch_size, FLAGS.sample_size)
        feed_dict = {
                placeholders['im_feat'] : im_feats,
                placeholders['sent_feat'] : sent_feats,
                placeholders['label'] : labels,
                placeholders['train_phase'] : False,
        }
        [recall_vals] = sess.run([recall], feed_dict = feed_dict)
        print('im2sent:', ' '.join(map(str, recall_vals[:3])),
              'sent2im:', ' '.join(map(str, recall_vals[3:])))


def main(_):
    # Load data.
    data_loader = DatasetLoader(FLAGS.image_feat_path, FLAGS.sent_feat_path, split='eval')
    num_ims, im_feat_dim = data_loader.im_feat_shape
    num_sents, sent_feat_dim = data_loader.sent_feat_shape

    # Setup placeholders for input variables.
    im_feat_plh = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, im_feat_dim])
    sent_feat_plh = tf.placeholder(tf.float32, shape=[FLAGS.batch_size * FLAGS.sample_size, sent_feat_dim])
    label_plh = tf.placeholder(tf.bool, shape=[FLAGS.batch_size * FLAGS.sample_size, FLAGS.batch_size])
    train_phase_plh = tf.placeholder(tf.bool)
    placeholders = {
        'im_feat' : im_feat_plh,
        'sent_feat' : sent_feat_plh,
        'label' : label_plh,
        'train_phase' : train_phase_plh,
    }

    # Setup testing operation.
    recall = setup_eval_model(im_feat_plh, sent_feat_plh, train_phase_plh, label_plh)

    # Setup checkpoint saver.
    saver = tf.train.Saver(save_relative_paths=True)

    # Periodically evaluate the latest checkpoint in the restore_dir directory,
    # unless a specific chekcpoint MetaGraph path is provided.
    while True:
        eval_once(data_loader, saver, placeholders, recall)
        if FLAGS.restore_path.endswith('.meta'):
            # Only evaluate the given checkpoint.
            break
        # Set the parameter to match the number of seconds to train 1 epoch.
        time.sleep(60)


if __name__ == '__main__':
    np.random.seed(0)
    tf.set_random_seed(0)

    parser = argparse.ArgumentParser()
    # Dataset and checkpoints.
    parser.add_argument('--image_feat_path', type=str, help='Path to the image feature mat file.')
    parser.add_argument('--sent_feat_path', type=str, help='Path to the sentence feature mat file.')
    parser.add_argument('--restore_path', type=str,
                        help='Directory for restoring the newest checkpoint or\
                              path to a restoring checkpoint MetaGraph file.')
    # Training parameters.
    parser.add_argument('--batch_size', type=int, default=1000, help='Batch size for evaluation.')
    parser.add_argument('--sample_size', type=int, default=5, help='Number of positive pair to sample.')

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
