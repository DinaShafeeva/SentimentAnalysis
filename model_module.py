import torch
import pickle
from transformers import TextClassificationPipeline
import numpy as np

model_name = 'model/second_model.sav'
model = pickle.load(open(model_name, 'rb'))
tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'DeepPavlov/rubert-base-cased')
pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=True)
labels = ['anger', 'sadness', 'neutral', 'joy', 'surprise', 'shame', 'disgust', 'fear']


def get_emotion_value(emotion):
    if emotion == 'anger':
        return -4
    if emotion == 'sadness':
        return -2
    if emotion == 'neutral':
        return 0
    if emotion == 'joy':
        return 2
    if emotion == 'surprise':
        return 5
    if emotion == 'shame':
        return -1
    if emotion == 'disgust':
        return -3.5
    if emotion == 'fear':
        return -3


class ModelModule:

    def predict_emotion_value(self, comment):
        piper = pipe(comment)
        list = [d['score'] for d in piper[0]]
        emotion = labels[np.argmax(list)]
        return get_emotion_value(emotion)

