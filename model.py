from tinygrad import Tensor, nn
from tinygrad.nn.state import get_state_dict,load_state_dict, safe_save, safe_load
import numpy as np

# Define the FNN model
class SimpleFNN():
    def __init__(self, dataset_size, output_size):
        self.fc1 = nn.Linear(dataset_size, 28)  # Input layer to hidden layer
        #self.ln1 = nn.LayerNorm(28)
        self.fc2 = nn.Linear(28, 14)  # Hidden layer to hidden layer
        #self.ln2 = nn.LayerNorm(14)
        self.fc3 = nn.Linear(14, output_size)   # Hidden layer to output layer with one neuron

    def forward(self, x):
        x = self.fc1(x)
        #x = self.ln1(x)
        x = self.fc2(x).relu()
        #x = self.ln2(x)
        x = self.fc3(x).sigmoid()
        return x  # Sigmoid activation for probability
    
    def save(self, filename):
        safe_save(get_state_dict(self), filename)

    # Load model function
    def load(self, filename):
        state_dict = safe_load(filename)
        load_state_dict(self, state_dict)
    
    def run(self, inputs):
        n = np.empty((1,6), dtype=np.float32)
        for i,name in enumerate(['filetype', 'platform', 'ppi_check','source', 'permissions', 'last_service']):
            n[0][i] = float(inputs[name])
        tinputs = Tensor(n)
        out = self.forward(tinputs).realize().numpy()
        return out.tolist()[0][0]

