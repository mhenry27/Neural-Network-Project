import sys
import numpy as np

class MLP: 
	def __init__(self, input_dim, hidden_units, learning_rate):
		#initialize weights and biases

		# w1 connects input -> hidden
		self.W1 = np.random.randn(input_dim, hidden_units) * 0.1
		self.b1 = np.zeros((1, hidden_units))

		#w2 connects hidden -> output
		self.W2 = np.random.randn(hidden_units, 1) * 0.1
		self.b2= np.zeros((1, 1))

		self.lr = learning_rate

	def forward(self, X):
		#hidden pre-activation
		#hidden activation using tanh
		#output pre-activation
		#output activation using sigmoid

		self.z1 = X @ self.W1 + self.b1
		self.h = np.tanh(self.z1)

		self.z2 = self.h @ self.W2 + self.b2
		y_hat = 1 / (1 + np.exp(-self.z2))

		return y_hat

	def compute_loss(self, y_hat, y):
		#binary cross entropy
		
		epsilon = 1e-8
		y_hat = np.clip(y_hat, epsilon, 1 - epsilon)
	
		loss = - np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

		return loss

	def backward(self, X, y, y_hat):
		#compute gradients with backprop
		#output error = y_hat - y

		batch_size = X.shape[0]

		dz2 = y_hat - y

		dW2 = self.h.T @ dz2 / batch_size
		db2 = np.mean(dz2, axis=0, keepdims = True)

		dh = dz2 @ self.W2.T
		dz1 = dh * (1 - self.h ** 2)

		dW1 = X.T @ dz1 / batch_size
		db1 = np.mean(dz1, axis=0, keepdims = True)

		return dW1, db1, dW2, db2


	def update_params(self, grads):
		#subtract learning_rate * gradient
		dW1, db1, dW2, db2 = grads

		self.W1 -= self.lr * dW1
		self.b1 -= self.lr * db1
		self.W2 -= self.lr * dW2
		self.b2 -= self.lr * db2

	def train_epoch(self, X, y, batch_size=1):
		#shuffle data
		#loop through minibatches
		#forward
		#backward
		#update weights

		perm = np.random.permutation(len(X))
		X, y = X[perm], y[perm]

		for i in range(0, len(X), batch_size):
			X_batch = X[i:i+batch_size]
			y_batch = y[i:i+batch_size]

			y_hat = self.forward(X_batch)
			grads = self.backward(X_batch, y_batch, y_hat)
			self.update_params(grads)

	def predict(self, X):
		#return 0/1 predictions using threshold 0.5

		y_hat = self.forward(X)
		return (y_hat >= .5).astype(int)

	def evaluate(self, X, y):
		#compute loss and F1 score
		y_hat = self.forward(X)
		loss = self.compute_loss(y_hat, y)
		y_pred = (y_hat >= 0.5).astype(int)
		f1 = f1_score(y, y_pred)
		return loss, f1


def load_data(path):
	# read text file
	# features are all columns except last
	# labels are last colomn

	data = np.loadtxt(path)
	X = data[:, :-1]
	y = data[:, -1].reshape(-1, 1)

	return X, y

def standardize_train(X):
	#comput mean and std from training data
	
	mean = np.mean(X, axis=0)
	std = np.std(X, axis=0)
	
	std[std == 0] = 1

	X_norm = (X - mean) / std

	return X_norm, mean, std

def apply_standardization(X, mean, std):
	#apply same mean/std to dev or test
	return (X - mean) / std

def f1_score(y_true, y_pred):
	# compute TP, FP, FN
	# precision = TP / (TP + FP)
	# recall = TP / (TP + FN)
	# F1 = 2PR / (P + R)

	TP = np.sum((y_pred == 1) & (y_true == 1))
	FP = np.sum((y_pred == 1) & (y_true == 0))
	FN = np.sum((y_pred == 0) & (y_true == 1))

	precision = TP / (TP + FP) if (TP + FP) > 0 else 0
	recall = TP / (TP + FN) if (TP + FN) > 0 else 0

	return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

def main():
	# check command line arguments 
	# python3 main.py <dataset> <hidden_units> <epochs> <learning_rate>
	
	# load dataset
	# standardize features 
	# initialize model
	# train for epochs
	# print loss and F1 each epoch
	
	if len(sys.argv) != 5:
	        print("Usage: python3 main.py <dataset> <hidden_units> <epochs> <learning_rate>")
        	sys.exit(1)

	dataset_path = sys.argv[1]
	hidden_units = int(sys.argv[2])
	epochs = int(sys.argv[3])
	learning_rate = float(sys.argv[4])

	X_train, y_train = load_data(dataset_path)

    	# Standardize using this dataset as the training set
	if "xor" not in dataset_path and "linear" not in dataset_path:
		if "magic" in dataset_path:
			X_train, mean, std = standardize_train(X_train)

	input_dim = X_train.shape[1]

	model = MLP(input_dim, hidden_units, learning_rate)

	train_f1s = []
	train_losses = []

	for epoch in range(epochs):
		model.train_epoch(X_train, y_train)

		train_loss, train_f1 = model.evaluate(X_train, y_train)

		train_losses.append(train_loss)
		train_f1s.append(train_f1)

		print(f"Epoch {epoch + 1}: Loss={train_loss:.6f}, F1={train_f1:.4f}")

	np.save(f"f1_h{hidden_units}.npy", train_f1s)
	np.save(f"loss_h{hidden_units}.npy", train_losses)


if __name__ == "__main__":
	main()	
