import torch
from tqdm import tqdm
import torch.nn as nn
# from torchmetrics import JaccardIndex
from torchmetrics.classification import BinaryJaccardIndex


class MonuSegModel:
	def __init__(self, model, device, n_classes=1, weights=None, criterion=None, lr=None, epochs=None):
		self.model = model
		self.criterion = criterion
		self.device = device
		self.epochs = epochs
		self.config_model(n_classes, weights)
		if lr is not None:
			self.optimizer = torch.optim.RMSprop(
				self.model.parameters(), lr=lr) # TODO: Try other optimizers
		else:
			self.optimizer = None

		self.train_losses = []
		self.valid_losses = []
		self.valid_accuracies = []
		self.mean_ious = []

	def config_model(self, n_classes, weights):
		if weights is not None:
			self.model.load_state_dict(torch.load(weights))
		self.model.to(self.device)

	def train(self, dataloader):
		self.model.train()

		epoch_losses = []

		data_size = 0
		avg_loss = 0

		for i, (images, masks) in tqdm(enumerate(dataloader)):
			images = images.to(self.device)
			masks = masks.to(self.device)

			self.optimizer.zero_grad() # Why do we need to do this? https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch
			outputs = self.model(images)
			loss = self.criterion(outputs, masks)
			loss.backward()
			self.optimizer.step()

			data_size += images.size(0)
			# avg_loss += loss.item() * images.size(0)
			avg_loss = (avg_loss * data_size + loss.item()) / (data_size + images.size(0)) # May be wrong

			epoch_losses.append(avg_loss)

		return avg_loss, epoch_losses


	def evaluate(self, dataloader):
		self.model.eval()

		# Do not record gradients
		with torch.no_grad():
			epoch_losses = []
			epoch_accuracy = []
			epoch_iou = []

			data_size = 0
			avg_loss = 0
			avg_accuracy = 0
			mean_iou = 0

			jaccard = BinaryJaccardIndex(threshold=0.35).to(self.device)

			for i, (images, masks) in tqdm(enumerate(dataloader)):
				images = images.to(self.device)

				masks[masks <= 0.65] = 0
				masks[masks > 0.65] = 1

				masks = masks.to(self.device)

				outputs = self.model(images)
				loss = self.criterion(outputs, masks)
				avg_loss = (avg_loss * data_size + loss.item()) / (data_size + images.size(0))
				avg_accuracy = (avg_accuracy * data_size + jaccard(outputs, masks)) / (data_size + images.size(0)) # May be wrong
				mean_iou += jaccard(outputs, masks).item() # May be wrong

				epoch_losses.append(avg_loss)
				epoch_accuracy.append(avg_accuracy)
				# epoch_iou.append(mean_iou)

				data_size += images.size(0)

			mean_iou /= len(dataloader)
			print(f'Mean IOU: {mean_iou}')

			return avg_loss, avg_accuracy, mean_iou, epoch_losses, epoch_accuracy

	def fit(self, train_dataloader, valid_dataloader):
		best_measure = -1
		best_epoch = -1

		if self.epochs is None or self.criterion is None or self.optimizer is None:
			raise ValueError(
				"Missing parameters \"epochs/criterion/optimizer\"")

		for epoch in range(self.epochs):
			print('-' * 10)
			print('Epoch {}/{}'.format(epoch, self.epochs - 1))
			print('-' * 10)

			# train and evaluate model performance
			train_loss, _ = self.train(train_dataloader)
			valid_loss, measure, mean_iou, epoch_losses, epoch_accuracy = self.evaluate(valid_dataloader)

			print(f"\nTrain Loss: {train_loss}")
			print(f"Valid Loss: {valid_loss}")
			print(f'Measure: {measure.item()}')

			# save metrics
			self.train_losses.append(float(train_loss))
			self.valid_losses.append(float(valid_loss))
			self.valid_accuracies.append(float(measure))
			self.mean_ious.append(float(mean_iou))

			if measure > best_measure:
				print(f'Updating best measure: {best_measure} -> {measure}')
				best_epoch = epoch
				best_weights = self.model.state_dict()
				best_measure = measure

		return best_epoch, best_measure, best_weights

	def predict_batches(self, dataloader):
		self.model.eval()

		# Do not record gradients
		with torch.no_grad():
			predictions = []
			for i, (images, masks) in tqdm(enumerate(dataloader)):
				images = images.to(self.device)
				outputs = self.model(images)
				predictions.append(outputs.cpu().numpy())

			return np.vstack(predictions)




