import torch.nn as nn

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

