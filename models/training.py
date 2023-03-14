from keras.utils import Sequence
from tensorflow import keras
import tensorflow as tf
from keras import layers
import keras_tuner
from pathlib import Path
import numpy as np
import pandas as pd
import h5py
from keras.models import Model
from PIL import Image
import os

from transformers import LayoutLMv3Config, LayoutLMModel, LayoutLMv3Tokenizer, LayoutLMTokenizer


class Generator(Sequence):

    def __init__(self, path_data: str, path_cache_folder: str = None, max_seq_length: int = 100,
                 input_img_size: tuple = (512, 512)):
        self.path_data = Path(path_data)
        self.max_seq_length = max_seq_length
        self.path_cache_folder = path_cache_folder + ".hdf5"
        self.input_img_size = input_img_size
        tokenizer = LayoutLMTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")

        all_path_tsv = list(self.path_data.glob("**/*.tsv"))
        self.page_ids = np.sort([path.stem for path in all_path_tsv])
        if os.path.isfile(self.path_cache_folder):
            os.remove(self.path_cache_folder)
        self.hf = h5py.File(self.path_cache_folder, 'a')
        df = create_df(self.page_ids)

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

    def __len__(self):
        """
        :returns number of batches per epoch
        """
        return len(self.page_ids)

    def __getitem__(self, idx):
        """
        receives call from keras (index) and grabs corresponding data batch
        :param index:
        :return:
        """
        page_id = self.page_ids[idx]
        self.hf = h5py.File(self.path_cache, 'r')
        # ocr
        input_ids = tf.convert_to_tensor(np.array(self.hf[page_id + '/input_ids']))
        bbox = tf.convert_to_tensor(np.array(self.hf[page_id + '/bbox']))
        attention_mask = tf.convert_to_tensor(np.array(self.hf[page_id + '/attention_mask']))
        token_type_ids = tf.convert_to_tensor(np.array(self.hf[page_id + '/token_type_ids']))
        ocr_input = {"input_ids": input_ids, 'bbox': bbox, 'attention_mask': attention_mask,
                     "token_type_ids": token_type_ids}
        # image
        image = tf.convert_to_tensor(np.array(self.hf[page_id + '/image']))

        self.hf.close()

        # get label
        label = get_label(page_id)

        return ocr_input, image, label, page_id


def generator(page_ids, path_cache):
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
    #img = tf.image.rgb_to_grayscale(img)
    #img = tf.image.grayscale_to_rgb(img)
    img = img[tf.newaxis, ..., tf.newaxis]
    img = tf.image.resize(img, input_img_size)
    #layer = layers.Normalization(mean=[0.485, 0.456, 0.406], variance=[np.square(0.229), np.square(0.224), np.square(0.225)])
    #img = layer(img)
    return img

def get_label(page_id):
    split = page_id.split('-')
    if len(split) == 1:  # single page doc
        return 1
    else:
        if int(split[1]) == 0:  # first page
            return 1
        else:  # non-first page
            return 0


def build_model(hp):
    model = keras.Sequential()
    model.add(layers.Flatten())
    for i in range(1, hp.Int("num_layers", 5, 10)):
        model.add(
            layers.Conv1D(
                kernel_size=hp.Int("kernel", min_value=3, max_value=6, step=1),
                strides=1,
                padding="valid",
                data_format="channels_last",
                dilation_rate=1,
                groups=1,
                activation='relu',
                use_bias=True
            )
        )
    model.add(layers.Dense(2, activation="softmax"))
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"],
    )
    return model


def create_df(page_ids):
    pages = np.array([(id.split('-') if len(id.split('-')) == 2 else [id, '0']) for id in page_ids])
    df = pd.DataFrame(pages, columns=['doc_id', 'page_num'])
    return df


def extend_box(words, boxes, tokenizer):
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


def tokenize_from_ocr(path_tsv, tokenizer, max_seq_length):
    '''
        Generate layoutlm input vectors from OCR
    '''

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
        #input_ids[-1] = 102
        attention_mask = attention_mask[:max_seq_length]
        token_type_ids = token_type_ids[:max_seq_length]
    bbox = token_boxes

    return input_ids, bbox, attention_mask, token_type_ids


def main():
   # configuration = LayoutLMv3Config()
   # layoutlm = LayoutLMModel.from_pretrained("microsoft/layoutlm-base-uncased")

   # resnet = keras.applications.ResNet50(weights="imagenet")
   # resnet = Model(resnet.input, resnet.layers[-2].output)

    traingen = Generator("../../TABME/data/train1/", "../../TABME/cache/train1")

    ds_counter = tf.data.Dataset.from_generator(lambda: generator(traingen.page_ids, traingen.path_cache_folder), output_types=({"input_ids": tf.int64, 'bbox': tf.float64, 'attention_mask': tf.float64,
                     "token_type_ids": tf.int64}, tf.int64, tf.int64, tf.string))

    for count_batch in ds_counter.batch(64).take(-1):
        inputs, image, labels, pageids = count_batch
        #print(tf.squeeze(image))
        print(labels)

# build_model(keras_tuner.HyperParameters())
# tuner = keras_tuner.Hyperband(build_model,
#           objective='val_accuracy',
#   max_epochs=10,
#   factor=3,
#   directory='my_dir',
#  project_name='intro_to_kt')


if __name__ == "__main__":
    main()
