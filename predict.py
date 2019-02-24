# -*- coding: utf-8 -*-

import tensorflow as tf
from config import config
from data_helper import load_json, padding


class Predict():
    def __init__(self, config, model_path='./runs/1550838941/checkpoints/model-1800', word_to_index='./vocabs/word_to_index.json',
                 index_to_label='./vocabs/index_to_label.json'):
        self.word_to_index = load_json(word_to_index)
        self.index_to_label = load_json(index_to_label)

        graph = tf.Graph()
        with graph.as_default():
            session_conf = tf.ConfigProto(
                allow_soft_placement=config['allow_soft_placement'],
                log_device_placement=config['log_device_placement'])
            self.sess = tf.Session(config=session_conf)
            with self.sess.as_default():
                # Load the saved meta graph and restore variables
                saver = tf.train.import_meta_graph("{}.meta".format(model_path))
                saver.restore(self.sess, model_path)

                # Get the placeholders from the graph by name
                self.input_x = graph.get_operation_by_name("input_x").outputs[0]

                self.dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

                # Tensors we want to evaluate
                self.predictions = graph.get_operation_by_name("output/predictions").outputs[0]

    def predict(self, list_str):
        input_x = padding(list_str, None, config, self.word_to_index, None)
        feed_dict = {
            self.input_x: input_x,
            self.dropout_keep_prob: 1.0
        }
        predictions = self.sess.run(self.predictions, feed_dict=feed_dict)
        return [self.index_to_label[str(idx)] for idx in predictions]


if __name__ == '__main__':
    prediction = Predict(config)
    result = prediction.predict(["我以前的互助怎么查不到", "升级后等待期又从升级的时候开始算吗", "会员50周岁以后呢", "我已满50周岁是否能参加"])
    print(result)
