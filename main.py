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
        self.intents_responses = {}

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

    def bag_of_words(self, words):
        return [1 if word in words else 0 for word in self.vocabulary]

    def parse_intents(self):
        lemmetizer = nltk.WordNetLemmatizer()

        if os.path.exists(self.intents_path):
            with open(self.intents_path, "r") as f:
                intents_data = json.load(f)

            for intent in intents_data["intents"]:
                if intent["tag"] not in self.intents:
                    self.intents.append(intent["tag"])
                    self.intents_responses.append(intent["tag"]) = intent["responses"]
                
                for pattern in intent["patterns"]:
                    pattern_words = self.tokenize_and_lemmatize(pattern)
                    self.vocabulary.extend(pattern_words)
                    self.document.append((pattern_words, intent["tag"]))

                self.vocabulary = sorted(set(self.vocabulary))

    def prepare_data(self):
        bags = []
        indices = []

        for document in self.documents:
            words = document[0]
            bag = self.bag_of_words(words)
            
            intent_index = self.intents.index(document[1])
            bags.append(bag)
            indices.append(intent_index)
        
        self.X = np.array(bags)
        self.Y = np.array(indices)

    def train_model(self, batch_size, lr, epochs):
        X_tensor = torch.tensor(self.X, dtype=torch.float32)
        Y_tensor = torch.tensor(self.Y, dtype=torch.long)

        dataset = TensorDataset(X_tensor, Y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model = ChatBotModel(self.X.shape[1], len(self.intents))
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        for epoch in range(epochs):