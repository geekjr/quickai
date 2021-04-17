"""
from quickai import TextFineTuning
"""

from pathlib import Path
import json
import os
import re
import numpy as np
import tensorflow as tf
from transformers import TFDistilBertForSequenceClassification, TFDistilBertForTokenClassification, \
    DistilBertTokenizerFast, TFDistilBertForQuestionAnswering
from sklearn.model_selection import train_test_split


def read_wnut(file_path):
    file_path = Path(file_path)

    raw_text = file_path.read_text().strip()
    raw_docs = re.split(r'\n\t?\n', raw_text)
    token_docs = []
    tag_docs = []
    for doc in raw_docs:
        tokens = []
        tags = []
        for line in doc.split('\n'):
            token, tag = line.split('\t')
            tokens.append(token)
            tags.append(tag)
        token_docs.append(tokens)
        tag_docs.append(tags)

    return token_docs, tag_docs


def encode_tags(tags, encodings, tag2id):
    labels = [[tag2id[tag] for tag in doc] for doc in tags]
    encoded_labels = []
    for doc_labels, doc_offset in zip(labels, encodings.offset_mapping):
        doc_enc_labels = np.ones(len(doc_offset), dtype=int) * -100
        arr_offset = np.array(doc_offset)

        doc_enc_labels[(arr_offset[:, 0] == 0) & (
                arr_offset[:, 1] != 0)] = doc_labels
        encoded_labels.append(doc_enc_labels.tolist())

    return encoded_labels


class TextFineTuning:
    def __init__(
            self,
            path,
            save,
            model_type,
            classes=None,
            epochs=3,
            batch_size=8):
        self.path = path
        self.save = save
        self.model_type = model_type
        self.batch_size = batch_size
        self.classes = classes
        self.epochs = epochs
        self.use()

    def read_split(self, split_dir):
        split_dir = Path(split_dir)
        texts = []
        labels = []
        for label_dir in self.classes:
            for text_file in (split_dir / label_dir).iterdir():
                texts.append(text_file.read_text(encoding="utf8"))
                labels.append(0 if label_dir == "neg" else 1)

        return texts, labels

    def read_squad(path_to_use):
        path = Path(path_to_use)
        with open(path, 'rb') as _:
            squad_dict = json.load(_)

        contexts = []
        questions = []
        answers = []
        for group in squad_dict['data']:
            for passage in group['paragraphs']:
                context = passage['context']
                for _ in passage['qas']:
                    question = _['question']
                    for answer in _['answers']:
                        contexts.append(context)
                        questions.append(question)
                        answers.append(answer)

        return contexts, questions, answers

    def add_end_idx(answers, contexts):
        for answer, context in zip(answers, contexts):
            gold_text = answer['text']
            start_idx = answer['answer_start']
            end_idx = start_idx + len(gold_text)

            if context[start_idx:end_idx] == gold_text:
                answer['answer_end'] = end_idx
            elif context[start_idx - 1:end_idx - 1] == gold_text:
                answer['answer_start'] = start_idx - 1
                answer['answer_end'] = end_idx - 1
            elif context[start_idx - 2:end_idx - 2] == gold_text:
                answer['answer_start'] = start_idx - 2
                answer['answer_end'] = end_idx - 2

    def add_token_positions(encodings, answers):
        tokenizer = DistilBertTokenizerFast.from_pretrained(
            'distilbert-base-uncased')
        start_positions = []
        end_positions = []
        for i in range(len(answers)):
            start_positions.append(
                encodings.char_to_token(
                    i, answers[i]['answer_start']))
            end_positions.append(
                encodings.char_to_token(
                    i, answers[i]['answer_end'] - 1))

            if start_positions[-1] is None:
                start_positions[-1] = tokenizer.model_max_length
            if end_positions[-1] is None:
                end_positions[-1] = tokenizer.model_max_length

        encodings.update({'start_positions': start_positions,
                          'end_positions': end_positions})

    def use(self):
        if self.model_type == "classification":
            train_texts, train_labels = self.read_split(f"{self.path}/train")

            train_texts, val_texts, train_labels, val_labels = train_test_split(
                train_texts, train_labels, test_size=0.2)
            tokenizer = DistilBertTokenizerFast.from_pretrained(
                'distilbert-base-uncased')
            train_encodings = tokenizer(
                train_texts, truncation=True, padding=True)
            val_encodings = tokenizer(val_texts, truncation=True, padding=True)

            train_dataset = tf.data.Dataset.from_tensor_slices((
                dict(train_encodings),
                train_labels
            ))
            val_dataset = tf.data.Dataset.from_tensor_slices((
                dict(val_encodings),
                val_labels
            ))

            model = TFDistilBertForSequenceClassification.from_pretrained(
                "distilbert-base-uncased")

        if self.model_type == "token_classification":
            texts, tags = self.read_wnut(self.path)

            train_texts, val_texts, train_tags, val_tags = train_test_split(
                texts, tags, test_size=.2)

            unique_tags = set(tag for doc in tags for tag in doc)
            tag2id = {tag: id for id, tag in enumerate(unique_tags)}

            tokenizer = DistilBertTokenizerFast.from_pretrained(
                'distilbert-base-cased')
            train_encodings = tokenizer(
                train_texts,
                is_split_into_words=True,
                return_offsets_mapping=True,
                padding=True,
                truncation=True)
            val_encodings = tokenizer(
                val_texts,
                is_split_into_words=True,
                return_offsets_mapping=True,
                padding=True,
                truncation=True)

            train_labels = self.encode_tags(
                train_tags, train_encodings, tag2id)
            val_labels = self.encode_tags(val_tags, val_encodings, tag2id)

            train_encodings.pop("offset_mapping")
            val_encodings.pop("offset_mapping")

            train_dataset = tf.data.Dataset.from_tensor_slices((
                dict(train_encodings),
                train_labels
            ))
            val_dataset = tf.data.Dataset.from_tensor_slices((
                dict(val_encodings),
                val_labels
            ))

            model = TFDistilBertForTokenClassification.from_pretrained(
                'distilbert-base-cased', num_labels=len(unique_tags))

        if self.model_type == "q+a":
            train_contexts, train_questions, train_answers = self.read_squad(
                f"{self.path}/train-v2.0.json")
            val_contexts, val_questions, val_answers = self.read_squad(
                f"{self.path}/dev-v2.0.json")

            self.add_end_idx(train_answers, train_contexts)
            self.add_end_idx(val_answers, val_contexts)

            tokenizer = DistilBertTokenizerFast.from_pretrained(
                'distilbert-base-uncased')

            train_encodings = tokenizer(
                train_contexts,
                train_questions,
                truncation=True,
                padding=True)
            val_encodings = tokenizer(
                val_contexts,
                val_questions,
                truncation=True,
                padding=True)

            self.add_token_positions(train_encodings, train_answers)
            self.add_token_positions(val_encodings, val_answers)

            train_dataset = tf.data.Dataset.from_tensor_slices((
                {key: train_encodings[key] for key in ['input_ids', 'attention_mask']},
                {key: train_encodings[key] for key in ['start_positions', 'end_positions']}
            ))
            val_dataset = tf.data.Dataset.from_tensor_slices((
                {key: val_encodings[key] for key in ['input_ids', 'attention_mask']},
                {key: val_encodings[key] for key in ['start_positions', 'end_positions']}
            ))

            model = TFDistilBertForQuestionAnswering.from_pretrained(
                "distilbert-base-uncased")

            train_dataset = train_dataset.map(lambda x, y: (
                x, (y['start_positions'], y['end_positions'])))

            model.distilbert.return_dict = False

        optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
        model.compile(optimizer=optimizer, loss=model.compute_loss)
        model.fit(
            train_dataset.shuffle(1000).batch(
                self.batch_size),
            validation_data=val_dataset,
            epochs=self.epochs,
            batch_size=self.batch_size)
        try:
            os.mkdir(f"{self.save}")
            model.save_pretrained(self.save)
        except OSError:
            model.save_pretrained(self.save)
