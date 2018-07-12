import numpy as np 
import pandas as pd # Data Prcessing, I/O for csv
import seaborn as sns # Visualisation 
import matplotlib.pyplot as plt 

import torch
import torch.nn as nn
from torch.autograd import Variable

class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        out = self.linear(x)
        return out


# Load Data from https://www.kaggle.com/harlfoxem/housesalesprediction
housingData = pd.read_csv("./Data/kc_house_data.csv")

# Preliminary Analysis
print(housingData.head())
print(housingData.isnull().any())
print(housingData.dtypes)
with sns.plotting_context("paper") :
    g = sns.pairplot(housingData[['sqft_lot','sqft_above','sqft_basement','price','sqft_living','bedrooms']], 
                hue='bedrooms')
g.set(xticklabels=[])

plt.show()
plt.clf()

space = housingData['sqft_living']
price = housingData['price']

x_train = np.array(space).reshape(-1, 1).astype('float32')
y_train = np.array(price).reshape(-1,1).astype('float32')

# Linear regression model, Loss and optimizer
model = LinearRegressionModel(1, 1)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.00000001)  

# Train the model
num_epochs = 600
for epoch in range(num_epochs):
    # Convert numpy arrays to torch tensors
    inputs = Variable(torch.from_numpy(x_train))
    labels = Variable(torch.from_numpy(y_train))

    # Clearing the gradients w.r.t. parameters
    optimizer.zero_grad()

    # Forward pass
    # model = model.double() # torch complains that tensor type mismatch if not included
    outputs = model.forward(inputs)
    loss = criterion(outputs, labels)
    
    # Backward and optimize
    loss.backward() # Backpropagation
    optimizer.step() # Update of parameters
    
    if (epoch+1) % 5 == 0:
        print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

# Plot the graph
predicted = model.forward(torch.from_numpy(x_train)).detach().numpy()
plt.plot(x_train, y_train, 'ro', label='Original data')
plt.plot(x_train, predicted, label='Prediction')
plt.legend()
plt.show()

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')