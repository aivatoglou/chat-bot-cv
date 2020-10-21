import os
import json
import torch
import flask
import random
from flask import send_from_directory

from net import ChatNet
from helpers import tokenize, stem, bag_of_words

app = flask.Flask(__name__, template_folder='templates')
port = int(os.environ.get("PORT", 5000))

with open('intents.json', 'r') as json_data:
	intents = json.load(json_data)

data = torch.load("data.pth", map_location=torch.device('cpu'))

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

@app.route("/", methods=['GET', 'POST'])
def index():

	if flask.request.method == 'GET':

		return(flask.render_template('main.html', original_input="Giorgos", result="Hello, my name is Giorgos."))

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
			result = ["Can you try asking it a different way?", "I'm not trained for that exact question. Try asking another way?", "I don't understand."]
			return(flask.render_template('main.html', original_input="Giorgos", result=random.choice(result)))

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico', mimetype='image/vnd.microsoft.icon')

if __name__ == '__main__':

	app.run(host='0.0.0.0', port=port)
