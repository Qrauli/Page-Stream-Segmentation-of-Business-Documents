from training import image_to_tensor, tokenize_from_ocr, generator, Generator, create_df, convert_batch
from tensorflow import keras
import tensorflow as tf
from keras import layers
from pathlib import Path
import numpy as np
import pandas as pd
import h5py
from keras.models import Model
from transformers import TFLayoutLMModel, LayoutLMTokenizer
from PIL import Image
import os
import argparse


def main():
    parser = argparse.ArgumentParser(description='Evaluate the model')
    parser.add_argument('-c', '--clear', required=False, default=False, action="store_true",
                        help="clear created cache files")
    parser.add_argument('-b', '--batch', type=int, required=False, default=40, help="batch size used in training")

    args, unknown = parser.parse_known_args()

    clear = args.clear
    batchsize = args.batch

    model = keras.models.load_model('./best_model',
                                    custom_objects={"TFLayoutLMModel": TFLayoutLMModel})

    if clear:
        if os.path.exists("./cache/test.hdf5"):
            os.remove("./cache/test.hdf5")

    testgen = Generator("./test", "./cache/test")
    df = pd.read_csv("./test.csv")

    acc_metric = keras.metrics.CategoricalAccuracy()
    rec_metric = keras.metrics.Recall()
    pre_metric = keras.metrics.Precision()
    err_metric = keras.metrics.CategoricalCrossentropy()

    for folder_id in df.folder_id.unique():
        print(str(folder_id) + ' of ' + str(len(df.folder_id.unique())), end='\r')
        df_folder = df[df.folder_id == folder_id]
        stems = df_folder.stem.to_list()
        df_folder = create_df(stems)

        ds_counter_test = tf.data.Dataset.from_generator(lambda: generator(df_folder, testgen.path_cache_folder),
                                                         output_types=({"input_ids": tf.int32, 'bbox': tf.int32,
                                                                        'attention_mask': tf.int32,
                                                                        "token_type_ids": tf.int32}, tf.int64, tf.int64,
                                                                       tf.string))

        iterator = iter(ds_counter_test.batch(batchsize).take(-1))
        while True:
            try:
                count_batch = next(iterator)
                # Do something with the batch
                input_ids, bbox, attention_mask, token_type_ids, image, labels, pageids = convert_batch(count_batch)

                with tf.GradientTape() as tape:
                    logits = model([input_ids, bbox, attention_mask, token_type_ids, image], training=False)

                # fixing first and single element predicition
                if len(logits.get_shape().as_list()) > 1:
                    logits = tf.concat([tf.constant([[1, 0]], dtype=tf.float32), logits[1:]], axis=0)
                else:
                    logits = tf.constant([[1, 0]], dtype=tf.float32)

                # update metrics
                acc_metric.update_state(labels, logits)
                err_metric.update_state(labels, logits)
                logits = tf.math.argmax(logits, axis=1)
                labels = tf.math.argmax(labels, axis=1)
                rec_metric.update_state(labels, logits)
                pre_metric.update_state(labels, logits)

            except tf.errors.OutOfRangeError:
                break
            except StopIteration as e:
                break
            except ValueError as e:
                print("Single element prediction dicarded")
                continue
    test_err = err_metric.result()
    test_acc = acc_metric.result()
    test_pre = pre_metric.result()
    test_rec = rec_metric.result()
    test_f1 = 2 * (float(test_pre) * float(test_rec)) / (float(test_rec) + float(test_pre))
    print("Test Accuracy: %.4f" % (float(test_acc),))
    print("Test Precision: %.4f" % (float(test_rec),))
    print("Test Recall: %.4f" % (float(test_pre),))
    print("Test F1-Score: %.4f" % (float(test_f1),))
    print("Test error: %.4f" % (float(test_err),))


if __name__ == "__main__":
    main()