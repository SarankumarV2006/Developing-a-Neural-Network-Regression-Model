# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY
The objective of this experiment is to design, implement, and evaluate a Deep Learning–based Neural Network regression model to predict a continuous output variable from a given set of input features. The task is to preprocess the data, construct a neural network regression architecture, train the model using backpropagation and gradient descent, and evaluate its performance using appropriate regression metrics such as Mean Squared Error (MSE), Mean Absolute Error (MAE), and R² score.

## Neural Network Model
<img width="947" height="617" alt="image" src="https://github.com/user-attachments/assets/6cb63e3a-3361-4482-ba3a-784921477450" />


## DESIGN STEPS
### STEP 1: 

Create your dataset in a Google sheet with one numeric input and one numeric output.

### STEP 2: 

Split the dataset into training and testing

### STEP 3: 

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4: 

Build the Neural Network Model and compile the model.

### STEP 5: 

Train the model with the training data.

### STEP 6: 

Plot the performance plot

### STEP 7: 

Evaluate the model with the testing data.

### STEP 8: 

Use the trained model to predict  for a new input value .

## PROGRAM

### Name: Sarankumar.V

### Register Number:212224220089

```
class NeuralNet(nn.Module):
  def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(1, 10)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(10, 1)
        self.history = {'loss': []}

  def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x



# Initialize the Model, Loss Function, and Optimizer
ai_brain = NeuralNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(ai_brain.parameters(), lr=0.01)


def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
    # Write your code here
    for epoch in range(epochs):
        # Forward pass
        outputs = ai_brain(X_train)
        loss = criterion(outputs, y_train)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ai_brain.history['loss'].append(loss.item())
        if epoch % 200 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')
```

### Dataset Information
### OUTPUT
<img width="245" height="297" alt="image" src="https://github.com/user-attachments/assets/d75f75d9-17b6-4f0b-9e28-3eaf3488c69b" />


### Training Loss Vs Iteration Plot
<img width="758" height="520" alt="image" src="https://github.com/user-attachments/assets/b4d795e4-2868-43cf-aeb1-05394adabc0a" />


### New Sample Data Prediction
<img width="356" height="45" alt="image" src="https://github.com/user-attachments/assets/8368b94e-ae00-4099-b2fd-517f565ebdcd" />


## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
