import os
import json
import random

import numpy as np  
import nltk

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# nltk.download('punkt_tab') run once for download the package

class ChatBotModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(ChatBotModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class ChatBotAssistant:
    def __init__(self, intents_path, function_mappings = None):
        self.model = None
        self.intents_path = intents_path

        self.document = []
        self.vocabulary = []
        self.intents = []
        self.intents_responses = []

        self.function_mappings = function_mappings

        self.X = None
        self.Y = None

    @staticmethod
    def tokenize_and_lemmatize(text):
        lammatizer = nltk.WordNetLemmatizer()
        words = nltk.word_tokenize(text)
        words = [lammatizer.lemmatize(word.lower()) for word in words]