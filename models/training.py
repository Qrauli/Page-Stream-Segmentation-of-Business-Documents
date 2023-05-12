from tensorflow import keras
import tensorflow as tf
from keras import layers
from pathlib import Path
import numpy as np
import pandas as pd
import h5py
from keras.models import Model
from PIL import Image
import os
import json
from transformers import TFLayoutLMModel, LayoutLMTokenizer
import pickle
from classification_models.tfkeras import Classifiers
import argparse
import sys


class Generator:
    """class storing and creating training data with tokenizers and h5py cache files"""

    def __init__(self, path_data: str, path_cache_folder: str = None, max_seq_length: int = 100,
                 input_img_size: tuple = (224, 224)):
        self.path_data = Path(path_data)
        self.max_seq_length = max_seq_length
        self.path_cache_folder = path_cache_folder + ".hdf5"
        self.input_img_size = input_img_size
        tokenizer = LayoutLMTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")

        all_path_tsv = list(self.path_data.glob("**/*.tsv"))
        self.page_ids = np.sort([path.stem for path in all_path_tsv])
        if os.path.isfile(self.path_cache_folder):
            return
        self.hf = h5py.File(self.path_cache_folder, 'a')

        # save each page in h5py file
        for page_id in self.page_ids:
            #tokenize ocr data
            file_id = page_id.split('-')[0]
            path_tsv = path_data + "/" + file_id + "/" + f"{page_id}.tsv"
            input_ids, bbox, attention_mask, token_type_ids = tokenize_from_ocr(path_tsv, tokenizer, 100)
            group = self.hf.create_group(page_id)
            group.create_dataset('input_ids', data=input_ids, compression="gzip")
            group.create_dataset('bbox', data=bbox, compression="gzip")
            group.create_dataset('attention_mask', data=attention_mask, compression="gzip")
            group.create_dataset('token_type_ids', data=token_type_ids, compression="gzip")
            path_img = path_data + "/" + file_id + "/" + f"{page_id}.jpg"

            # preprocess image
            img = Image.open(path_img)
            image = image_to_tensor(tf.convert_to_tensor(img), input_img_size)
            group.create_dataset('image', data=image, compression="gzip")
        self.hf.close()


def generator(df_pages, path_cache):
    """generator returning extracted data of single elements"""
    i = 0
    while i < len(df_pages):
        page_id = df_pages.iloc[i]["page_id"]
        hf = h5py.File(path_cache, 'r')

        # ocr
        input_ids = tf.convert_to_tensor(np.array(hf[page_id + '/input_ids']))
        bbox = tf.convert_to_tensor(np.array(hf[page_id + '/bbox']))
        attention_mask = tf.convert_to_tensor(np.array(hf[page_id + '/attention_mask']))
        token_type_ids = tf.convert_to_tensor(np.array(hf[page_id + '/token_type_ids']))
        ocr_input = {"input_ids": input_ids, 'bbox': bbox, 'attention_mask': attention_mask,
                     "token_type_ids": token_type_ids}
        # image
        image = tf.convert_to_tensor(np.array(hf[page_id + '/image']))

        hf.close()

        # get label
        label = get_label(page_id)

        yield ocr_input, image, label, page_id
        i += 1


def create_df(page_ids):
    """Create pandas dataframe for pages"""
    pages = np.array([(id.split('-') + [id] if len(id.split('-')) == 2 else [id, '0', id]) for id in page_ids])
    df = pd.DataFrame(pages, columns=['doc_id', 'page_num', 'page_id'])
    return df


def shuffle_df(df):
    """Shuffle IDs while retaining the same page order within the document"""
    df_id = df.value_counts(subset='doc_id').to_frame(name='num_pages')
    # reassign document
    df_id = df_id.sample(frac=1)
    df_id['order'] = range(len(df_id))
    df['order'] = df.doc_id.apply(lambda id: df_id.loc[id, 'order'])
    # must choose a stable sort to retain the page order within the same document
    df = df.sort_values(by='order', kind='mergesort')
    # reassign page order
    df['order'] = range(len(df))
    return df


def image_to_tensor(img, input_img_size):
    """converts grayscale image to tensorflow tensor with resnet specifications"""
    img = img[..., tf.newaxis]
    img = tf.image.grayscale_to_rgb(img)
    img = tf.image.resize(img, input_img_size)
    return img


def get_label(page_id):
    """returns the label for a training image based on its name"""
    split = page_id.split('-')
    if len(split) == 1:  # single page doc
        return 1
    else:
        if int(split[1]) == 0:  # first page
            return 1
        else:  # non-first page
            return 0


def extend_box(words, boxes, tokenizer):
    """extend box size"""
    token_boxes = []
    tokenizable_words = words
    j = 0  # index for tokenizable words
    for i in range(len(words)):
        word, box = words[i], boxes[i]
        try:
            word_tokens = tokenizer.tokenize(word)
            token_boxes.extend([box] * len(word_tokens))
            j += 1
        except:
            tokenizable_words = np.delete(tokenizable_words, j)

    return tokenizable_words, token_boxes


def tokenize_from_ocr(path_tsv, tokenizer, max_seq_length, dataframe=None):
    """Generate layoutlm input vectors from OCR: method taken and modified from TABME project"""
    df_ocr_page = dataframe
    if dataframe is None:
        df_ocr_page = pd.read_csv(path_tsv, sep='\t')

    # If the there are lots of boxes, select only the boxes with high confidence. Will not filter if the number of
    # boxes is too few.
    df_high_conf = df_ocr_page[df_ocr_page.conf > 50]
    if len(df_high_conf) >= max_seq_length:
        df_ocr_page = df_high_conf

    words = df_ocr_page['text'].to_numpy()
    heights = df_ocr_page['height'].to_numpy()
    widths = df_ocr_page['width'].to_numpy()
    lefts = df_ocr_page['left'].to_numpy()
    tops = df_ocr_page['top'].to_numpy()
    boxes = np.column_stack((lefts, tops, widths + lefts, heights + tops))  # [[x0, y0, x1, y1], ...]

    words, token_boxes = extend_box(words, boxes, tokenizer)

    # Add bounding boxes of cls + sep tokens
    token_boxes = [[0, 0, 0, 0]] + token_boxes + [[1000, 1000, 1000, 1000]]

    encoding = tokenizer(' '.join(words.tolist()), return_tensors="pt", padding=True, pad_to_multiple_of=max_seq_length)
    input_ids = tf.squeeze(encoding["input_ids"], 0)
    attention_mask = tf.squeeze(encoding["attention_mask"], 0)
    token_type_ids = tf.squeeze(encoding["token_type_ids"], 0)

    # Pad/cut token_boxes with [0, 0, 0, 0] or cut the elements beyond {max_seq_length}
    pad_box = [0, 0, 0, 0]
    if len(token_boxes) <= max_seq_length:
        token_boxes = tf.convert_to_tensor(np.array(token_boxes + (max_seq_length - len(token_boxes)) * [pad_box]))
    else:
        token_boxes = tf.convert_to_tensor(np.array(token_boxes[:max_seq_length - 1] + [[1000, 1000, 1000, 1000]]))
        input_ids = input_ids[:max_seq_length]
        temp_input_ids = input_ids.numpy()
        temp_input_ids[-1] = 102
        input_ids = tf.convert_to_tensor(temp_input_ids)
        attention_mask = attention_mask[:max_seq_length]
        token_type_ids = token_type_ids[:max_seq_length]
    bbox = token_boxes

    return input_ids, bbox, attention_mask, token_type_ids


def create_model():
    """creates keras model based on pretrained models layoutlm and resnet combined with convolutional layers"""

    # pretrained models
    layoutlm = TFLayoutLMModel.from_pretrained("microsoft/layoutlm-base-uncased")
    ResNet18, preprocess_input = Classifiers.get('resnet18')
    resnet = ResNet18((224, 224, 3), weights='imagenet')

    # input layers
    input_ids_layer = tf.keras.Input(shape=(None,), dtype=tf.int32, name='input_ids')
    bbox_layer = tf.keras.Input(shape=(None, 4), dtype=tf.int32, name='bbox')
    token_type_ids_layer = tf.keras.Input(shape=(None,), dtype=tf.int32, name='token_type_ids')
    attention_mask_layer = tf.keras.Input(shape=(None,), dtype=tf.int32, name='attention_mask')

    layoutlm_layer = layoutlm(
        {'input_ids': input_ids_layer, 'bbox': bbox_layer, 'token_type_ids': token_type_ids_layer,
         'attention_mask': attention_mask_layer}).pooler_output
    layoutlm = tf.keras.Model(inputs=[input_ids_layer, bbox_layer, token_type_ids_layer, attention_mask_layer],
                              outputs=layoutlm_layer)

    resnet = Model(resnet.input, resnet.layers[-2].output)

    # convolutional layers
    sizes = [1067, 854, 641, 428, 215]
    x = layers.concatenate([resnet.layers[-1].output, layoutlm_layer], 1)
    x = keras.layers.Lambda(lambda y: tf.expand_dims(y, 0))(x)

    for i in range(1, 5):
        x = layers.Conv1D(filters=sizes[i], kernel_size=3, strides=1, padding="same", data_format="channels_last",
                          dilation_rate=1, groups=1, use_bias=True)(x)
        x = layers.ReLU()(x)
        x = layers.Dropout(0.2)(x)
    x = layers.Conv1D(filters=2, kernel_size=3, strides=1, padding="same", data_format="channels_last",
                      dilation_rate=1, groups=1, use_bias=True)(x)
    x = layers.ReLU()(x)
    x = keras.layers.Lambda(lambda y: tf.squeeze(y))(x)
    x = layers.Softmax()(x)

    model = keras.Model(
        inputs=[input_ids_layer, bbox_layer, token_type_ids_layer, attention_mask_layer, resnet.input],
        outputs=x
    )
    return model


def validate(model, validation_generator):
    """calculate validation error based on given model and generator returning validation data"""
    loss_fn = keras.losses.CategoricalCrossentropy(from_logits=False)
    iterator = iter(validation_generator.batch(40).take(-1))
    acc_metric = keras.metrics.CategoricalAccuracy()
    rec_metric = keras.metrics.Recall()
    acc1_metric = keras.metrics.Precision()
    err_metric = keras.metrics.CategoricalCrossentropy()
    val_loss = 0
    while True:
        try:
            count_batch = next(iterator)
            # Do something with the batch
            input_ids, bbox, attention_mask, token_type_ids, image, labels, pageids = convert_batch(count_batch)
            with tf.GradientTape() as tape:
                logits = model([input_ids, bbox, attention_mask, token_type_ids, image], training=False)
                loss_value = loss_fn(labels, logits)
                val_loss += float(loss_value)
            acc_metric.update_state(labels, logits)
            err_metric.update_state(labels, logits)
            logits = tf.math.argmax(logits, axis=1)
            labels = tf.math.argmax(labels, axis=1)
            acc1_metric.update_state(labels, logits)
            rec_metric.update_state(labels, logits)

        except tf.errors.InvalidArgumentError:
            print("Skipping batch with malformed tensor.")
            continue
        except tf.errors.OutOfRangeError:
            break
        except StopIteration as e:
            break
    val_loss = err_metric.result()
    val_acc = acc_metric.result()
    val_acc1 = acc1_metric.result()
    val_rec = rec_metric.result()
    val_f1 = 2 * (float(val_acc1) * float(val_rec)) / (float(val_rec) + float(val_acc1))
    return float(val_loss), float(val_acc), float(val_acc1), float(val_rec), float(val_f1)


def convert_batch(batch):
    """converts created batch into data needed for model prediction and training"""
    inputs, image, labels, pageids = batch
    input_ids = inputs['input_ids']
    bbox = inputs['bbox']
    attention_mask = inputs['attention_mask']
    token_type_ids = inputs['token_type_ids']
    list1 = tf.unstack(labels)
    list1[0] = 1
    list0 = [1 if i == 0 else 0 for i in list1]
    labels = tf.stack(np.transpose([list1, list0]))
    return input_ids, bbox, attention_mask, token_type_ids, image, labels, pageids


def main():
    parser = argparse.ArgumentParser(description='Train the model')
    parser.add_argument('-c', '--clear', required=False, default=False, action="store_true",
                        help="clear created cache files")
    parser.add_argument('-r', '--resume', required=False, default=False, action="store_true",
                        help="resume last training run")
    parser.add_argument('-b', '--batch', type=int, required=False, default=40, help="batch size used in training")

    args, unknown = parser.parse_known_args()

    clear = args.clear
    continue_train = args.resume
    batch_size = args.batch

    path_model = "my_model.json"
    data_continue = []
    if os.path.exists(path_model) and continue_train:
        print("reusing previous train data")
        with open(path_model, "r") as file:
            data_continue = json.load(file)
    else:
        data = {"epoch": 0, "best_val": 0, "early_stopping_cnt": 0}
        if os.path.exists("./my_model.json"):
            os.remove("./my_model.json")
        with open(path_model, "x") as file:
            json.dump(data, file)

    if continue_train:
        model = keras.models.load_model('./my_model',
                                        custom_objects={"TFLayoutLMModel": TFLayoutLMModel})
    else:
        model = create_model()

    if clear:
        if os.path.exists("./train.hdf5"):
            os.remove("./train.hdf5")
        if os.path.exists("./validation.hdf5"):
            os.remove("./validation.hdf5")

    traingen = Generator("./train", "./train")
    valgen = Generator("./val", "./validation")
    dftrain = create_df(traingen.page_ids)
    dfval = create_df(valgen.page_ids)

    ds_counter_val = tf.data.Dataset.from_generator(lambda: generator(dfval, valgen.path_cache_folder),
                                                    output_types=({"input_ids": tf.int32, 'bbox': tf.int32,
                                                                   'attention_mask': tf.int32,
                                                                   "token_type_ids": tf.int32}, tf.int64, tf.int64,
                                                                  tf.string))

    loss_fn = keras.losses.CategoricalCrossentropy(from_logits=False)
    optimizer = keras.optimizers.Adam(learning_rate=1e-5)
    if continue_train:
        with open('./optimizerstate.pickle', 'rb') as handle:
            b = pickle.load(handle)
        optimizer = keras.optimizers.Adam.from_config(b)

    train_acc_metric = keras.metrics.CategoricalAccuracy()
    train_rec_metric = keras.metrics.Recall()
    train_acc1_metric = keras.metrics.Precision()
    train_err_metric = keras.metrics.CategoricalCrossentropy()

    epochs = 30
    min_val_loss = 1
    early_stopping_cnt = 0
    if continue_train:
        min_val_loss = data_continue["best_val"]
        early_stopping_cnt = data_continue["early_stopping_cnt"]
    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))
        if continue_train:
            if epoch < data_continue["epoch"]:
                continue

        dftrain_shuffled = shuffle_df(dftrain)

        ds_counter = tf.data.Dataset.from_generator(lambda: generator(dftrain_shuffled, traingen.path_cache_folder),
                                                    output_types=({"input_ids": tf.int32, 'bbox': tf.int32,
                                                                   'attention_mask': tf.int32,
                                                                   "token_type_ids": tf.int32}, tf.int64, tf.int64,
                                                                  tf.string))

        counter = 0
        iterator = iter(ds_counter.batch(batch_size).take(-1))
        data = []
        with open(path_model, "r") as file:
            data = json.load(file)
        data["epoch"] = epoch
        with open(path_model, "w") as file:
            json.dump(data, file)

        while True:
            try:
                count_batch = next(iterator)

                # Do something with the batch
                counter = counter + 1
                data = []
                with open(path_model, "r") as file:
                    data = json.load(file)
                data["batch"] = counter
                with open(path_model, "w") as file:
                    json.dump(data, file)
                input_ids, bbox, attention_mask, token_type_ids, image, labels, pageids = convert_batch(count_batch)

                with tf.GradientTape() as tape:
                    logits = model([input_ids, bbox, attention_mask, token_type_ids, image], training=True)
                    loss_value = loss_fn(labels, logits)

                grads = tape.gradient(loss_value, model.trainable_weights)
                optimizer.apply_gradients(zip(grads, model.trainable_weights))

                # Update training metric.
                train_acc_metric.update_state(labels, logits)
                train_err_metric.update_state(labels, logits)
                logits = tf.math.argmax(logits, axis=1)
                labels = tf.math.argmax(labels, axis=1)
                train_acc1_metric.update_state(labels, logits)
                train_rec_metric.update_state(labels, logits)
                # Log every 200 batches.
                if counter % 200 == 0:
                    print("Training loss (for one batch) at step %d: %.4f" % (counter, float(loss_value)))
                    print("Seen so far: %d samples" % ((counter + 1) * 30))

                if counter % 1000 == 0:
                    val_loss, val_acc, val_acc1, val_rec, val_f1 = validate(model, ds_counter_val)
                    print("Validation loss at step %d: %.4f" % (counter, float(val_loss)))
                    print("Validation acc at step %d: %.4f" % (counter, float(val_acc)))
                    print("Validation precision at step %d: %.4f" % (counter, float(val_rec)))
                    print("Validation recall at step %d: %.4f" % (counter, float(val_acc1)))
                    print("Validation f1 at step %d: %.4f" % (counter, float(val_f1)))
                    if val_loss < min_val_loss:
                        print("new best model!")
                        early_stopping_cnt = 0
                        model.save("best_model")
                        min_val_loss = val_loss
                        data = []
                        with open(path_model, "r") as file:
                            data = json.load(file)
                        data["best_val"] = min_val_loss
                        with open(path_model, "w") as file:
                            json.dump(data, file)
                    else:
                        early_stopping_cnt += 1
                        data = []
                        with open(path_model, "r") as file:
                            data = json.load(file)
                        data["early_stopping_cnt"] = early_stopping_cnt
                        with open(path_model, "w") as file:
                            json.dump(data, file)
                        if early_stopping_cnt > 9:
                            print("Early Stopping: No improvement over last 5 epochs")
                            sys.exit()

                # Display metrics at the end of each epoch.
            except tf.errors.InvalidArgumentError:
                print("Skipping batch with malformed tensor.")
                continue
            except tf.errors.OutOfRangeError:
                print("End of dataset.")
                break
            except StopIteration as e:
                print("End of dataset.")
                break

        train_acc = train_acc_metric.result()
        train_acc1 = train_rec_metric.result()
        train_rec = train_acc1_metric.result()
        train_err = train_err_metric.result()
        train_f1 = 2 * (float(train_acc1) * float(train_rec)) / (float(train_rec) + float(train_acc1))
        model.save("my_model")
        with open('./optimizerstate.pickle', 'wb') as file:
            pickle.dump(optimizer.get_config(), file, protocol=pickle.HIGHEST_PROTOCOL)
        print("Training Accuracy over epoch: %.4f" % (float(train_acc),))
        print("Training Precision over epoch: %.4f" % (float(train_acc1),))
        print("Training Recall over epoch: %.4f" % (float(train_rec),))
        print("Training F1-Score over epoch: %.4f" % (float(train_f1),))
        print("Training error over epoch: %.4f" % (float(train_err),))
        train_acc_metric.reset_state()
        train_acc1_metric.reset_state()
        train_rec_metric.reset_state()
        train_err_metric.reset_state()


if __name__ == "__main__":
    main()