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


def main():
    model = keras.models.load_model('./my_model',
                                    custom_objects={"TFLayoutLMModel": TFLayoutLMModel})

    testgen = Generator("../../TABME/data/test/", "../../TABME/cache/test")
    df = pd.read_csv("../../TABME/predictions/test.csv")

    loss_fn = keras.losses.CategoricalCrossentropy(from_logits=False)
    acc_metric = keras.metrics.CategoricalAccuracy()
    rec_metric = keras.metrics.Recall()
    acc1_metric = keras.metrics.Precision()
    err_metric = keras.metrics.CategoricalCrossentropy()
    # iterator = iter(generator.batch(40).take(-1))

    for folder_id in df.folder_id.unique():
        df_folder = df[df.folder_id == folder_id]
        stems = df_folder.stem.to_list()
        df_folder = create_df(stems)

        ds_counter_test = tf.data.Dataset.from_generator(lambda: generator(df_folder, testgen.path_cache_folder),
                                                        output_types=({"input_ids": tf.int32, 'bbox': tf.int32,
                                                                      'attention_mask': tf.int32,
                                                                      "token_type_ids": tf.int32}, tf.int64, tf.int64,
                                                                      tf.string))

        iterator = iter(ds_counter_test.batch(40).take(-1))
        while True:
            try:
                count_batch = next(iterator)
                # Do something with the batch
                input_ids, bbox, attention_mask, token_type_ids, image, labels, pageids = convert_batch(count_batch)
                # outputs = model([input_ids, bbox, attention_mask, token_type_ids, image])
                with tf.GradientTape() as tape:
                    logits = model([input_ids, bbox, attention_mask, token_type_ids, image], training=False)

                acc_metric.update_state(labels, logits)
                err_metric.update_state(labels, logits)
                rec_metric.update_state(labels, logits)
                acc1_metric.update_state(labels, logits)

            except tf.errors.InvalidArgumentError:
                print("Skipping batch with malformed tensor.")
                continue
            except tf.errors.OutOfRangeError:
                break
            except StopIteration as e:
                break
            except Exception as e:
                continue
    test_err = err_metric.result()
    test_acc = acc_metric.result()
    test_acc1 = acc1_metric.result()
    test_rec = rec_metric.result()
    test_f1 = 2 * (float(test_acc1) * float(test_rec)) / (float(test_rec) + float(test_acc1))
    print("Test Accuracy: %.4f" % (float(test_acc),))
    print("Test Precision: %.4f" % (float(test_acc1),))
    print("Test Recall: %.4f" % (float(test_rec),))
    print("Test F1-Score: %.4f" % (float(test_f1),))
    print("Test error: %.4f" % (float(test_err),))



if __name__ == "__main__":
    main()