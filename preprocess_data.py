#%cd drive/MyDrive/ECS289_final
import tensorflow as tf

# Scikit-learn includes many helpful utilities
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import re
import numpy as np
import pandas as pd
import os
import time
import json
import collections
import operator

from glob import glob
from PIL import Image
import pickle
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

import numpy as np
from tensorflow import reshape
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, Embedding, LSTM, Activation,ZeroPadding1D,Conv1D
from tqdm import tqdm


class PrepareData():
    def __init__(self):
        print('Creating an object...')
        self.PATH = 'dataset/val2014/'
        self.annotation_file = 'dataset/val2014/v2_mscoco_val2014_annotations.json'
        self.question_file = 'dataset/val2014/v2_OpenEnded_mscoco_val2014_questions.json'
        self.all_data = os.listdir(self.PATH[:-1])
        
        
    # read the json file
    def parse_answers(self):
        print("Parsing answers...")
        with open(self.annotation_file, 'r') as f:
            annotations = json.load(f)

        # storing the captions and the image name in vectors
        self.all_answers = []
        self.all_answers_qids = []
        self.all_img_name_vector = []
        self.filter_list = []

        for index, annot in tqdm(enumerate(annotations['annotations'])):
            if ' ' not in annot['multiple_choice_answer']:
                caption = '<start> ' + annot['multiple_choice_answer'] + ' <end>'
                image_id = annot['image_id']
                question_id = annot['question_id']
                #full_coco_image_path = self.PATH + 'COCO_train2014_' + '%012d.jpg_dense' % (image_id)
                full_coco_image_path = self.PATH + 'COCO_val2014_' + '%012d.jpg' % (image_id)
                
                self.all_img_name_vector.append(full_coco_image_path)
                self.all_answers.append(caption)
                self.all_answers_qids.append(question_id)
            else:
                filter_list.append(index)
                
            
    def parse_questions(self):
        # read the json file
        print("Parsing questions...")
        with open(self.question_file, 'r') as f:
            questions = json.load(f)

        # storing the captions and the image name in vectors
        self.question_ids =[]
        self.all_questions = []
        self.all_img_name_vector_2 = []

        for index, annot in tqdm(enumerate(questions['questions'])):
            if index not in self.filter_list:
                caption = '<start> ' + annot['question'] + ' <end>'
                image_id = annot['image_id']
                full_coco_image_path = self.PATH + 'COCO_val2014_' + '%012d.jpg' % (image_id)
    
                self.all_img_name_vector_2.append(full_coco_image_path)
                self.all_questions.append(caption)
                self.question_ids.append(annot['question_id'])
        
      
    def shuffle_extract_data(self, num_examples=-1):
        print("Extracting data...")
        self.train_answers, self.train_questions, self.img_name_vector = shuffle(self.all_answers, self.all_questions,
                                              self.all_img_name_vector,
                                              random_state=1)

        # selecting the first 3000 captions from the shuffled set
        if num_examples:
            self.train_answers = self.train_answers[:num_examples]
            self.train_questions = self.train_questions[:num_examples]
            self.img_name_vector = self.img_name_vector[:num_examples]

            
    def load_image(self, image_path):
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (299, 299))
        img = tf.keras.applications.inception_v3.preprocess_input(img)
        return img, image_path

    
    def extract_image_features(self, spatial_features = True):
        
        print("Extracting image feature...")
        image_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')

        # Extract features from intermediate layer of image_model
        layer_index = -1
        if spatial_features:
            new_input = image_model.input
            hidden_layer = image_model.layers[layer_index].output # 
            image_features_extract_model = tf.keras.Model(new_input, hidden_layer)
        else:
            image_features_extract_model = image_model

        # getting the unique images
        encode_train = sorted(set(self.img_name_vector))

        # feel free to change the batch_size according to your system configuration
        image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
        image_dataset = image_dataset.map(self.load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(16)

        for img, path in tqdm(image_dataset):
            batch_features = image_features_extract_model(img)
            # unroll image to sequence
            # batch_features = tf.reshape(batch_features, (batch_features.shape[0], -1, batch_features.shape[3])) 
            # print(batch_features.shape)
            
            for bf, p in zip(batch_features, path):
                if p not in self.all_data:
                    path_of_feature = p.numpy().decode("utf-8")
                    np.save(path_of_feature, bf.numpy())

                
    # This will find the maximum length of any question in our dataset
    def calc_max_length(self, tensor):
        return max(len(t) for t in tensor)


    # choosing the top 10000 words from the vocabulary, words other than top 10000 will be treated as unkown
    def create_question_vector(self, top_k_words=1000):
        print("Creating question vector...")
        self.question_tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k_words,
                                                          oov_token="<unk>",
                                                          filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
        self.question_tokenizer.fit_on_texts(self.train_questions)

        self.ques_vocab = self.question_tokenizer.word_index
        self.question_tokenizer.word_index['<pad>'] = 0
        self.question_tokenizer.index_word[0] = '<pad>'

        # creating the tokenized vectors
        train_question_seqs = self.question_tokenizer.texts_to_sequences(self.train_questions)

        # padding each vector to the max_length of the captions
        # if the max_length parameter is not provided, pad_sequences calculates that automatically
        self.question_vector = tf.keras.preprocessing.sequence.pad_sequences(train_question_seqs, padding='post')

        # calculating the max_length
        # used to store the attention weights
        self.max_q = self.calc_max_length(train_question_seqs)


    def create_answer_vector(self, top_k_words=1000):
        print("Creating answer vector...")
        self.answer_tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k_words,
                                                          oov_token="<unk>",
                                                          filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
        self.answer_tokenizer.fit_on_texts(self.train_answers)

        self.ans_vocab = self.answer_tokenizer.word_index
        self.answer_tokenizer.word_index['<pad>'] = 0
        self.answer_tokenizer.index_word[0] = '<pad>'

        # creating the tokenized vectors
        train_answer_seqs = self.answer_tokenizer.texts_to_sequences(self.train_answers)

        # padding each vector to the max_length of the captions
        # if the max_length parameter is not provided, pad_sequences calculates that automatically
        self.answer_vector = tf.keras.preprocessing.sequence.pad_sequences(train_answer_seqs, padding='post')

        # calculating the max_length
        # used to store the attention weights
        self.max_a = self.calc_max_length(train_answer_seqs)

        
    # loading the numpy files
    def map_func(self, img_name, cap, ans):
      img_tensor = np.load(img_name.decode('utf-8')+'.npy')
      return img_name, img_tensor, cap,ans


    def get_dataset(self, BATCH_SIZE, BUFFER_SIZE, features_shape, attention_features_shape):
        print("Creating dataset")
        img_name_train, img_name_val, question_train, question_val,answer_train, answer_val  = train_test_split(self.img_name_vector,
                                                                    self.question_vector,
                                                                    self.answer_vector,
                                                                    test_size=0.2,
                                                                    random_state=0)

        dataset = tf.data.Dataset.from_tensor_slices((img_name_train, question_train.astype(np.float32), answer_train.astype(np.float32)))

        # using map to load the numpy files in parallel
        dataset = dataset.map(lambda item1, item2, item3: tf.numpy_function(self.map_func, [item1, item2, item3], [tf.string, tf.float32, tf.float32, tf.float32]),
                  num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # shuffling and batching
        dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        print(dataset)
        test_dataset = tf.data.Dataset.from_tensor_slices((img_name_val, question_val.astype(np.float32), answer_val.astype(np.float32)))

        # using map to load the numpy files in parallel
        test_dataset = test_dataset.map(lambda item1, item2, item3: tf.numpy_function(
                  self.map_func, [item1, item2, item3], [tf.string, tf.float32, tf.float32, tf.float32]),
                  num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # shuffling and batching
        test_dataset = test_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
        test_dataset = test_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return dataset, test_dataset, self.ques_vocab, self.ans_vocab
