# block#1
seed = 5

# numpy==1.24.1
import numpy as np
# torch==2.1.0
import torch

torch.manual_seed(seed)
np.random.seed(seed)
# endblock

# block#2
n_timesteps, n_assets = 2000, 5
lookback, gap, horizon = 40, 2, 10

n_samples = n_timesteps - lookback - horizon - gap + 1
split_ix = int(n_samples * 0.8)
indices_train = list(range(split_ix))
indices_test = list(range(split_ix + lookback + horizon, n_samples))
# endblock

# block#3
# Synthetic sinusoidal returns
from utils.data_utils import sin_single

returns = np.array(
    [
        sin_single(
            n_timesteps,
            freq = 1 / np.random.randint(3, lookback),
            amplitude = 0.01,
            phase = np.random.randint(0, lookback)
        ) for _ in range(n_assets)
    ]
).T

# We also add some noise.
returns += np.random.normal(scale = 0.002, size = returns.shape)

# Mean returns
print(f'Mean returns: {round(sum(np.mean(returns, axis = 0)), 5)}')
# Mean returns: 0.00013

# Plot returns
# matplotlib==3.7.2
import matplotlib.pyplot as plt

plt.title('Returns')
plt.plot(returns[:100, 0:(n_assets - 1)])
plt.show()
# endblock

# block#4
# Creating sliding window sets
X_list, y_list = [], []

for i in range(lookback, n_timesteps - horizon - gap + 1):
    X_list.append(returns[i - lookback: i, :])
    y_list.append(returns[i + gap: i + gap + horizon, :])

X = np.stack(X_list, axis = 0)[:, None, ...]
y = np.stack(y_list, axis = 0)[:, None, ...]

print(f'X: {X.shape}, y: {y.shape}')
# endblock

# block#5
# deepdow==0.2.2
from deepdow.data import InRAMDataset, RigidDataLoader, prepare_standard_scaler, Scale

means, stds = prepare_standard_scaler(X, indices = indices_train)
dataset = InRAMDataset(X, y, transform = Scale(means, stds))

batch_size = 512
dataloader_train = RigidDataLoader(
    dataset,
    indices = indices_train,
    batch_size = batch_size
)

dataloader_test = RigidDataLoader(
    dataset,
    indices = indices_test,
    batch_size = batch_size
)
# endblock

# block#6
from model.fully_connected_trivial_net import FullyConnectedTrivialNet

network = FullyConnectedTrivialNet(n_assets, lookback)
network = network.train()
# endblock

# block#7
from deepdow.losses import SharpeRatio
from torch.optim import Adam

loss = SharpeRatio()
optimizer = Adam(network.parameters(), amsgrad = True)

train_epochs = 10
print('Training...')
for epoch in range(train_epochs):
    error_list = []
    for batch_idx, batch in enumerate(dataloader_train):
        X_batch, y_batch, timestamps, asset_names = batch

        X_batch = X_batch.float()
        y_batch = y_batch.float()

        weights = network(X_batch)
        error = loss(weights, y_batch).mean()
        error_list.append(error.item())
        optimizer.zero_grad()
        error.backward()
        optimizer.step()
    print(f'Epoch {epoch} | Error: {round(np.mean(error_list), 5)}')
# endblock

# block#8
network = network.eval()

# Weights for the test set
weights_list = []
timestamps_list = []
for X_batch, _, timestamps, _ in dataloader_test:
    X_batch = X_batch.float()
    weights = network(X_batch).detach().numpy()

    weights_list.append(weights)
    timestamps_list.extend(timestamps)

weights = np.concatenate(weights_list, axis = 0)
asset_names = [dataloader_test.dataset.asset_names[asset_ix] for asset_ix in dataloader_test.asset_ixs]

# pandas==1.5.3
import pandas as pd

weights_df = pd.DataFrame(
    weights,
    index = timestamps_list,
    columns = asset_names
)
weights_df.sort_index(inplace = True)
# endblock

# block#9
# Multiply the returns by the weights to get the portfolio returns.
# returns_df - empty DataFrame for storing portfolio returns with same index as weights_df
model_returns_df, uniform_returns_df = pd.DataFrame(), pd.DataFrame()
model_returns_df.index, uniform_returns_df.index = weights_df.index, weights_df.index

uniform_weights = np.ones(n_assets) / n_assets
for idx in range(indices_test[0], indices_test[-1] - gap - horizon, horizon):
    i_model_returns = (np.array(weights_df.loc[idx].tolist()) * returns[idx + gap: idx + gap + horizon]).sum(axis = 1)
    i_uniform_returns = (uniform_weights * returns[idx + gap: idx + gap + horizon]).sum(axis = 1)
    # Store the returns in the DataFrame to range [idx + gap: idx + gap + horizon]
    model_returns_df.loc[idx + gap: idx + gap + len(i_model_returns) - 1, 'return'] = i_model_returns
    uniform_returns_df.loc[idx + gap: idx + gap + len(i_uniform_returns) - 1, 'return'] = i_uniform_returns

# Drop the NaN values from the DataFrame.
model_returns_df.dropna(inplace = True)
uniform_returns_df.dropna(inplace = True)
# endblock

# block#10
import os
# QuantStats==0.0.62
import quantstats as qs

model_balance_df = (model_returns_df['return'] + 1).cumprod()
uniform_balance_df = (uniform_returns_df['return'] + 1).cumprod()

# Adding index to the balance_df as days from 2000-01-01
model_balance_df.index = pd.date_range(start = '2000-01-01', periods = len(model_balance_df))
uniform_balance_df.index = pd.date_range(start = '2000-01-01', periods = len(uniform_balance_df))

current_dir = os.path.dirname(os.path.abspath(__file__))
report_path = os.path.join(current_dir, '../output/synth_variational_allocation/')
qs.reports.html(
    model_balance_df,
    title = 'Model Results for Synthetic Variational Allocation',
    output = report_path + 'model_results.html'
)
qs.reports.html(
    uniform_balance_df,
    title = 'Uniform Results for Synthetic Variational Allocation',
    output = report_path + 'uniform_results.html'
)
# endblock
