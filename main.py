import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader

from train import train
from data import GDataset
from unet import UNet

if __name__ == '__main__':

	trans = transforms.Compose([
    	transforms.ToTensor(),
	])

	train_set = GDataset(2000, transform = trans)
	val_set = GDataset(200, transform = trans)

	image_datasets = {
    	'train': train_set, 'val': val_set
	}

	batch_size = 10

	data_loader = {
    	'train': DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0),
    	'val': DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=0)
	}

	num_of_classes = 6

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	print('Device is {}'.format(device))

	model = UNet(num_of_classes).to(device)

	optimizer = optim.Adam(model.parameters(), lr=1e-4)
	scheduler = lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)

	model = train(model, optimizer, scheduler, data_loader, device, 40)

	torch.save(model.state_dict(), 'unet_model.pt')