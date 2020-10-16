import random
import json
import torch
import torch.nn as nn
from nltk.stem.porter import PorterStemmer
import numpy as np
import nltk
import flask
import os

app = flask.Flask(__name__, template_folder='templates')

port = int(os.environ.get("PORT", 5001))

class ChatNet(nn.Module):
	def __init__(self, input_size, hidden_size1, hidden_size2, num_classes):
		super(ChatNet, self).__init__()
		self.l1 = nn.Linear(input_size, hidden_size1) 
		self.l2 = nn.Linear(hidden_size1, hidden_size1) 
		self.l3 = nn.Linear(hidden_size1, hidden_size2)
		self.l4 = nn.Linear(hidden_size2, num_classes)
		self.relu = nn.ReLU()
	
	def forward(self, x):
		out = self.l1(x)
		out = self.relu(out)
		out = self.l2(out)
		out = self.relu(out)
		out = self.l3(out)
		out = self.relu(out)
		out = self.l4(out)
		# no activation and no softmax at the end
		return out

def tokenize(sentence):
	return nltk.word_tokenize(sentence)

def stem(word):
	return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, words):
	sentence_words = [stem(word) for word in tokenized_sentence]
	bag = np.zeros(len(words), dtype=np.float32)
	for idx, w in enumerate(words):
		if w in sentence_words: 
			bag[idx] = 1
	return bag

stemmer = PorterStemmer()

with open('intents.json', 'r') as json_data:
	intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size1 = data["hidden_size1"]
hidden_size2 = data["hidden_size2"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = ChatNet(input_size, hidden_size1, hidden_size2, output_size)
model.load_state_dict(model_state)
model.eval()

print("Let's chat! (type 'quit' to exit)")

@app.route("/", methods=['GET', 'POST'])
def index():

	if flask.request.method == 'GET':

		return(flask.render_template('main.html', original_input="Giorgos", result="Hello!"))


	if flask.request.method == 'POST':

		sentence = flask.request.form['sentence']

		sentence = tokenize(sentence)
		X = bag_of_words(sentence, all_words)
		X = X.reshape(1, X.shape[0])
		X = torch.from_numpy(X)

		output = model(X)
		_, predicted = torch.max(output, dim=1)

		tag = tags[predicted.item()]

		probs = torch.softmax(output, dim=1)
		prob = probs[0][predicted.item()]
		if prob.item() > 0.75:
			for intent in intents['intents']:
				if tag == intent["tag"]:
					result = random.choice(intent['responses'])
					return(flask.render_template('main.html', original_input="Giorgos", result=result))
		else:
			return(flask.render_template('main.html', original_input="Giorgos", result="I don't understand..."))

if __name__ == '__main__':

	app.run(host='0.0.0.0', port=port)
