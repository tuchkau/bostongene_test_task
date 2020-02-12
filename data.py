from torch.utils.data import Dataset
from utils import generate_random_data

class GDataset(Dataset):
	def __init__(self, count, transform = None):
		self.inputs, self.labels = generate_random_data(192, 192, count=count)
		self.transform = transform

	def __getitem__(self, idx):
		input_img = self.inputs[idx]
		label_img = self.labels[idx]

		if self.transform:
			input_img = self.transform(input_img)

		return [input_img, label_img]

	def __len__(self):
		return len(self.inputs)

