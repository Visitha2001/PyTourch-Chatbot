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

# download if package not exist
def download_if_not_exists(package, path="./nltk_data"):
    try:
        nltk.data.find(f"corpora/{package}")
    except LookupError:
        nltk.download(package, download_dir=path)
        nltk.data.path.append(path)

# Ensure required packages
download_if_not_exists("wordnet")
download_if_not_exists("omw-1.4")
download_if_not_exists("punkt")

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

        return words

# chatbot = ChatBotAssistant("intents.json")
# print(chatbot.tokenize_and_lemmatize("Hello world how are you, i am programming in python today."))

    @staticmethod
    def bag_of_words(words, vocabulary):
        return [1 if word in words else 0 for word in vocabulary]

    def parse_intents(self):
        lemmetizer = nltk.WordNetLemmatizer()

        if os.path.exists(self.intents_path):
            with open(self.intents_path, "r") as f:
                intents_data = json.load(f)

            intents = []
            intents_responses = []
            vocabulary = []
            documents = []

            for intent in intents_data["intents"]:
                if intent["tag"] not in intents:
                    intents.append(intent["tag"])
                    intents_responses.append(intent["tag"]) = intent["responses"]
                
                for pattern in intent["patterns"]:
                    pattern_words = self.tokenize_and_lemmatize(pattern)
                    vocabulary.extend(pattern_words)
                    documents.append((pattern_words, intent["tag"]))