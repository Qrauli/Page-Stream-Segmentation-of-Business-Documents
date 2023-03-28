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
import time
import json
from transformers import TFLayoutLMModel, LayoutLMTokenizer


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
            file_id = page_id.split('-')[0]
            path_tsv = path_data + "/" + file_id + "/" + f"{page_id}.tsv"
            input_ids, bbox, attention_mask, token_type_ids = tokenize_from_ocr(path_tsv, tokenizer, 100)
            group = self.hf.create_group(page_id)
            group.create_dataset('input_ids', data=input_ids, compression="gzip")
            group.create_dataset('bbox', data=bbox, compression="gzip")
            group.create_dataset('attention_mask', data=attention_mask, compression="gzip")
            group.create_dataset('token_type_ids', data=token_type_ids, compression="gzip")
            path_img = path_data + "/" + file_id + "/" + f"{page_id}.jpg"
            img = Image.open(path_img)
            image = image_to_tensor(tf.convert_to_tensor(img), input_img_size)
            group.create_dataset('image', data=image, compression="gzip")
        self.hf.close()


def generator(page_ids, path_cache):
    """generator returning extracted data of single elements"""
    i = 0
    while i < len(page_ids):
        page_id = page_ids[i]
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


# Image preprocessing
def image_to_tensor(img, input_img_size):
    """converts grayscale image to tensorflow tensor with resnet specifications"""
    # img = tf.image.rgb_to_grayscale(img)
    img = img[..., tf.newaxis]
    img = tf.image.grayscale_to_rgb(img)
    # print(img.get_shape())
    # img = img[tf.newaxis, ..., tf.newaxis]
    img = tf.image.resize(img, input_img_size)
    # img = tf.transpose(img, perm=[2, 0, 1])
    # layer = layers.Normalization(mean=[0.485, 0.456, 0.406], variance=[np.square(0.229), np.square(0.224),
    # np.square(0.225)]) img = layer(img)
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


#def build_model(hp):
#    model = keras.Sequential()
#    model.add(layers.Flatten())
#    for i in range(1, hp.Int("num_layers", 5, 10)):
#        model.add(
#            layers.Conv1D(
#                kernel_size=hp.Int("kernel", min_value=3, max_value=6, step=1),
#                strides=1,
#                padding="valid",
#                data_format="channels_last",
#                dilation_rate=1,
#                groups=1,
#                activation='relu',
#                use_bias=True
#            )
#        )
#    model.add(layers.Dense(2, activation="softmax"))
#    model.compile(
#        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"],
#    )
#    return model


#def create_df(page_ids):
#    pages = np.array([(page_id.split('-') if len(page_id.split('-')) == 2 else [page_id, '0']) for page_id in page_ids])
#    df = pd.DataFrame(pages, columns=['doc_id', 'page_num'])
#    return df


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
        # input_ids[-1] = 102
        attention_mask = attention_mask[:max_seq_length]
        token_type_ids = token_type_ids[:max_seq_length]
    bbox = token_boxes

    return input_ids, bbox, attention_mask, token_type_ids


def create_model():
    """creates keras model based on pretrained models layoutlm and resnet combined with convolutional layers"""
    layoutlm = TFLayoutLMModel.from_pretrained("microsoft/layoutlm-base-uncased")
    resnet = keras.applications.ResNet50V2(weights="imagenet")

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

    # sizes = [1280, 1067, 854, 641, 428, 215]
    # 2816
    sizes = [2347, 1878, 1409, 940, 471]
    x = layers.concatenate([resnet.layers[-1].output, layoutlm_layer], 1)
    # mypermute = lambda y: keras.backend.transpose(y)
    # x = keras.layers.Lambda(mypermute)(x)
    # x = keras.layers.Lambda(lambda x: tf.expand_dims(x, 2))(x)
    x = keras.layers.Lambda(lambda x: tf.expand_dims(x, 0))(x)

    # x = layers.Permute((2,1))(x)
    # x = keras.Input(shape=(sizes[0],))

    for i in range(1, 5):
        x = layers.Conv1D(filters=sizes[i], kernel_size=3, strides=1, padding="same", data_format="channels_last",
                          dilation_rate=1, groups=1, use_bias=True)(x)
        x = layers.ReLU()(x)
        x = layers.Dropout(0.2)(x)
    x = layers.Conv1D(filters=2, kernel_size=3, strides=1, padding="same", data_format="channels_last",
                      dilation_rate=1, groups=1, use_bias=True)(x)
    x = layers.ReLU()(x)
    x = keras.layers.Lambda(lambda x: tf.squeeze(x))(x)
    x = layers.Softmax()(x)

    # model.add(layers.Dense(2, activation="softmax"))
    # model.compile(
    # optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"],
    # )

    model = keras.Model(
        inputs=[input_ids_layer, bbox_layer, token_type_ids_layer, attention_mask_layer, resnet.input],
        outputs=x
    )
    return model


def validate(model, generator):
    """calculate validation error based on given model and generator returning validation data"""
    loss_fn = keras.losses.CategoricalCrossentropy(from_logits=False)
    iterator = iter(generator.batch(30).take(-1))
    val_loss = 0
    count = 0
    while True:
        try:
            count_batch = next(iterator)
            count += 1
            # Do something with the batch
            input_ids, bbox, attention_mask, token_type_ids, image, labels, pageids = convert_batch(count_batch)
            # outputs = model([input_ids, bbox, attention_mask, token_type_ids, image])
            with tf.GradientTape() as tape:
                logits = model([input_ids, bbox, attention_mask, token_type_ids, image], training=False)
                loss_value = loss_fn(labels, logits)
                val_loss += float(loss_value)
                print("loss: %.4f" % float(loss_value))

        except tf.errors.InvalidArgumentError:
            print("Skipping batch with malformed tensor.")
            continue
        except tf.errors.OutOfRangeError:
            break
        except StopIteration as e:
            break
    val_loss = val_loss / count
    return float(val_loss)


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
    path_model = "my_model.json"
    if os.path.exists(path_model):
        print("reusing previous train data")
    else:
        data = {"epoch": 0, "batch": 0, "best_val": 0, "early_stopping_cnt": 0}
        with open(path_model, "x") as file:
            json.dump(data, file)

    model = create_model()

    # configuration = LayoutLMv3Config()
    # layoutlm = TFLayoutLMModel.from_pretrained("microsoft/layoutlm-base-uncased")
    # layoutlm.build([None, 32, 32, 3])
    # inputs = tf.keras.Input(shape=(224, 224, 3))
    # layoutlm.call(inputs)

    # input_ids_layer = tf.keras.Input(shape=(None,), dtype=tf.int32, name='input_ids')
    # bbox_layer = tf.keras.Input(shape=(None, 4), dtype=tf.int32, name='bbox')
    # token_type_ids_layer = tf.keras.Input(shape=(None,), dtype=tf.int32, name='token_type_ids')
    # attention_mask_layer = tf.keras.Input(shape=(None,), dtype=tf.int32, name='attention_mask')

    # layoutlm_layer = layoutlm({'input_ids': input_ids_layer, 'bbox': bbox_layer, 'token_type_ids': token_type_ids_layer, 'attention_mask': attention_mask_layer}).pooler_output
    # layoutlm = tf.keras.Model(inputs=[input_ids_layer, bbox_layer, token_type_ids_layer, attention_mask_layer], outputs=layoutlm_layer)

    # print(layoutlm.summary())
    # layoutlm = Model(layoutlm.input, layoutlm.layers[-1].output)

    # resnet = keras.applications.ResNet50V2(weights="imagenet")
    # resnet = Model(resnet.input, resnet.layers[-2].output)

    traingen = Generator("../../TABME/data/train/", "../../TABME/cache/train")
    valgen = Generator("../../TABME/data/val/", "../../TABME/cache/validation")

    ds_counter = tf.data.Dataset.from_generator(lambda: generator(traingen.page_ids, traingen.path_cache_folder),
                                                output_types=({"input_ids": tf.int32, 'bbox': tf.int32,
                                                               'attention_mask': tf.int32,
                                                               "token_type_ids": tf.int32}, tf.int64, tf.int64,
                                                              tf.string))

    ds_counter_val = tf.data.Dataset.from_generator(lambda: generator(valgen.page_ids, valgen.path_cache_folder),
                                                    output_types=({"input_ids": tf.int32, 'bbox': tf.int32,
                                                                   'attention_mask': tf.int32,
                                                                   "token_type_ids": tf.int32}, tf.int64, tf.int64,
                                                                  tf.string))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])

    loss_fn = keras.losses.CategoricalCrossentropy(from_logits=False)
    optimizer = keras.optimizers.Adam(learning_rate=1e-5)
    train_acc_metric = keras.metrics.CategoricalAccuracy()

    # val_acc_metric = keras.metrics.SparseCategoricalAccuracy()

    epochs = 30
    train_acc_old = 0
    min_val_loss = 1
    early_stopping_cnt = 0
    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))
        start_time = time.time()
        counter = 0
        iterator = iter(ds_counter.batch(64).take(-1))
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
                # outputs = model([input_ids, bbox, attention_mask, token_type_ids, image])
                with tf.GradientTape() as tape:
                    logits = model([input_ids, bbox, attention_mask, token_type_ids, image], training=True)
                    loss_value = loss_fn(labels, logits)

                grads = tape.gradient(loss_value, model.trainable_weights)
                optimizer.apply_gradients(zip(grads, model.trainable_weights))

                # Update training metric.
                train_acc_metric.update_state(labels, logits)

                # Log every 200 batches.
                if counter % 50 == 0:
                    print(
                        "Training loss (for one batch) at step %d: %.4f"
                        % (counter, float(loss_value))
                    )
                    print("Seen so far: %d samples" % ((counter + 1) * 64))

                if counter % 1000 == 0:
                    val_loss = validate(model, ds_counter_val)
                    print(
                        "Validation loss at step %d: %.4f"
                        % (counter, float(val_loss))
                    )
                    if val_loss < min_val_loss:
                        model.save("my_model")
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
                        if early_stopping_cnt > 5:
                            print("Early Stopping: No improvement over last 5 models")
                            break

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
        # for count_batch in ds_counter.batch(64).take(-1):
        train_acc = train_acc_metric.result()
        if train_acc > train_acc_old:
            train_acc_old = train_acc
            model.save("my_model")

        print("Training acc over epoch: %.4f" % (float(train_acc),))

    # for count_batch in ds_counter.batch(64).take(-1):
    # inputs, image, labels, pageids = count_batch
    # print(tf.squeeze(image))
    # print(labels)
    # print(image)
    # imageinformation = resnet(image)
    # print(imageinformation.get_shape())
    # input_ids = inputs['input_ids']
    # bbox = inputs['bbox']
    # attention_mask = inputs['attention_mask']
    # token_type_ids = inputs['token_type_ids']
    # textinformation = layoutlm([input_ids, bbox, token_type_ids, attention_mask])
    # textinformation = layoutlm(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, token_type_ids=token_type_ids)['pooler_output']
    # print(textinformation.get_shape())
    # features = tf.concat([imageinformation, textinformation], 1)
    # print(features.get_shape())
    # outputs = model([input_ids, bbox, attention_mask, token_type_ids, image])
    # print(outputs)
    # print(outputs.get_shape())
    # print(output)
    # build_model(keras_tuner.HyperParameters())
    # tuner = keras_tuner.Hyperband(build_model,
    #           objective='val_accuracy',
    #   max_epochs=10,
    #   factor=3,
    #   directory='my_dir',
    #  project_name='intro_to_kt')


if __name__ == "__main__":
    main()
