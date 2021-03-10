import time
import os
import math
import numpy as np
import tensorflow as tf
import json
import base64
from PIL import Image, ImageDraw, ImageFont
from utils import CharRNNLM, SampleType, Word2Vec, RhymeWords


class config:
    model_dir = '/mnt/auto/model/poem'
    data_dir = '/mnt/auto/model/poem'
    length = 16
    max_prob = False
    seed = int(time.time())


class WritePoem():
    def __init__(self):
        self.args = config
        with open(os.path.join(self.args.model_dir, 'result.json'), 'r') as f:
            result = json.load(f)
        params = result['params']
        best_model = result['best_model']
        self.args.encoding = result['encoding'] if 'encoding' in result else 'utf-8'

        base_path = config.data_dir
        w2v_file = os.path.join(base_path, "vectors_poem.bin")
        self.w2v = Word2Vec(w2v_file)
        RhymeWords.read_rhyme_words(os.path.join(base_path, 'rhyme_words.txt'))
        if config.seed >= 0:
            np.random.seed(config.seed)
        self.sess = tf.Session()
        w2v_vocab_size = len(self.w2v.model.vocab)
        with tf.name_scope('evaluation'):
            self.model = CharRNNLM(is_training=False,
                                   w2v_model=self.w2v.model,
                                   vocab_size=w2v_vocab_size,
                                   infer=True,
                                   **params)
            saver = tf.train.Saver(name='model_saver')
            saver.restore(self.sess, best_model)

    def free_verse(self):
        sample = self.model.sample_seq(self.sess, 40, '[', sample_type=SampleType.weighted_sample)
        idx_end = sample.find(']')
        parts = sample.split('。')
        if len(parts) > 1:
            two_sentence_len = len(parts[0]) + len(parts[1])
            if idx_end < 0 or two_sentence_len < idx_end:
                return sample[1:two_sentence_len + 2]
        return sample[1:idx_end]

    @staticmethod
    def assemble(sample):
        if sample:
            parts = sample.split('。')
            if len(parts) > 1:
                return '{}。{}。'.format(parts[0][1:], parts[1][:len(parts[0])])
        return ''

    def rhyme_verse(self):
        gen_len = 20
        sample = self.model.sample_seq(self.sess,
                                       gen_len,
                                       start_text='[',
                                       sample_type=SampleType.weighted_sample)
        parts = sample.split('。')
        if len(parts) > 0:
            start = parts[0] + '。'
            rhyme_ref_word = start[-2]
            rhyme_seq = len(start) - 3
            sample = self.model.sample_seq(self.sess,
                                           gen_len,
                                           start,
                                           sample_type=SampleType.weighted_sample,
                                           rhyme_ref=rhyme_ref_word,
                                           rhyme_idx=rhyme_seq)
            return WritePoem.assemble(sample)
        return sample[1:]

    def hide_words(self, given_text):
        if (not given_text):
            return self.rhyme_verse()
        givens = ['', '']
        split_len = math.ceil(len(given_text) / 2)
        givens[0] = given_text[:split_len]
        givens[1] = given_text[split_len:]
        gen_len = 20
        sample = self.model.sample_seq(self.sess,
                                       gen_len,
                                       start_text='[',
                                       sample_type=SampleType.select_given,
                                       given=givens[0])
        parts = sample.split('。')
        if len(parts) > 0:
            start = parts[0] + '。'
            rhyme_ref_word = start[-2]
            rhyme_seq = len(start) - 3
            sample = self.model.sample_seq(self.sess,
                                           gen_len,
                                           start,
                                           sample_type=SampleType.select_given,
                                           given=givens[1],
                                           rhyme_ref=rhyme_ref_word,
                                           rhyme_idx=rhyme_seq)
            return WritePoem.assemble(sample)
        return sample[1:]

    def acrostic(self, given_text):
        start = ''
        rhyme_ref_word = ''
        rhyme_seq = 0
        for i in range(4):
            word = ''
            if i < len(given_text):
                word = given_text[i]
            if i == 0:
                start = '[' + word
            else:
                start += word
            before_idx = len(start)
            if i != 3:
                sample = self.model.sample_seq(self.sess,
                                               self.args.length,
                                               start,
                                               sample_type=SampleType.weighted_sample)
            else:
                if not word:
                    rhyme_seq += 1

                sample = self.model.sample_seq(self.sess,
                                               self.args.length,
                                               start,
                                               sample_type=SampleType.max_prob,
                                               rhyme_ref=rhyme_ref_word,
                                               rhyme_idx=rhyme_seq)
            sample = sample[before_idx:]
            idx1 = sample.find('，')
            idx2 = sample.find('。')
            min_idx = min(idx1, idx2)

            if min_idx == -1:
                if idx1 > -1:
                    min_idx = idx1
                else:
                    min_idx = idx2
            if min_idx > 0:
                start = '{}{}'.format(start, sample[:min_idx + 1])

                if i == 1:
                    rhyme_seq = min_idx - 1
                    rhyme_ref_word = sample[rhyme_seq]

        return WritePoem.assemble(start)


writer = WritePoem()


class Response:
    def __init__(self, start_response, response):
        self.start = start_response
        self.response = response

    def __iter__(self):
        status = '200'
        response_headers = [('Content-type', 'text/html')]
        self.start(status, response_headers)
        yield self.response.encode("utf-8")


def getPage():
    with open("index.html") as f:
        data = f.read()
    return data


def getPicture(text):
    font = ImageFont.truetype('font.ttf', 40)
    temp_base_pic = Image.open("love.jpg").convert("RGBA")
    draw = ImageDraw.Draw(temp_base_pic)
    wordList = text.split("。")
    for eve in range(0, len(wordList)):
        draw.text((300, (eve + 1) * 50 + 200), wordList[eve], (79, 79, 79), font=font)
    temp_base_pic.save("/tmp/test_output.png")
    with open("/tmp/test_output.png", "rb") as f:
        base64Data = str(base64.b64encode(f.read()), encoding='utf-8')
    return 'data:image/png;base64,' + base64Data


def handler(environ, start_response):
    path = environ['PATH_INFO'].replace("/api", "")

    if path == "/":
        return Response(start_response, getPage())
    else:
        try:
            request_body_size = int(environ.get('CONTENT_LENGTH', 0))
        except (ValueError):
            request_body_size = 0
        tempBody = environ['wsgi.input'].read(request_body_size).decode("utf-8")
        requestBody = json.loads(tempBody)

        if path == "/poem":
            return Response(start_response, writer.hide_words(requestBody.get("content", "我是江昱")))
