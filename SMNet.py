import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import MSELoss, L1Loss
import torch.optim as optim
from torch_points3d.applications.kpconv import KPConv
from torch_points3d.applications.pointnet2 import PointNet2
from torch_geometric.data import Batch, Data, Dataset, DataLoader
import torch_geometric.nn as geo_nn
from sklearn.metrics import r2_score
import time

class SMNetRegressor(nn.Module):
    def __init__(self, input_nc, num_classes, output_nc, output_dim):
        super(SMNetRegressor, self).__init__()
        self.kpconv = KPConv(architecture="unet", input_nc=input_nc, output_nc=num_classes, num_layers=4)
        self.pointnet2 = PointNet2(architecture="unet", input_nc=num_classes, num_layers=4, output_nc=output_nc)
        self.regressor = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            #nn.Dropout(p=0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            #nn.Dropout(p=0.3),
            nn.Linear(256, output_dim),
        )

    def forward(self, tdata, batch_size, num_points, num_classes):
        kpconv_output = self.kpconv(tdata)
        ttdata = Batch(batch=kpconv_output.batch.reshape(batch_size, num_points), pos=kpconv_output.pos.reshape(batch_size, num_points, 3), x=kpconv_output.x.reshape(batch_size, num_points, num_classes))
        pointnet2_output = self.pointnet2(ttdata)
        x_reshaped = pointnet2_output.x.permute(0, 2, 1)
        pooled_output = x_reshaped.mean(dim=1)
        x = self.regressor(pooled_output)
        return x

def normalize_point_clouds(point_clouds: np.ndarray) -> np.ndarray:
    centroids = np.mean(point_clouds, axis=1)
    point_clouds_centered = point_clouds-centroids[:, np.newaxis, :]
    max_norms = np.max(np.linalg.norm(point_clouds_centered, axis=2), axis=1)
    point_clouds_normalized = point_clouds_centered/max_norms[:, np.newaxis, np.newaxis]
    return point_clouds_normalized

csv_name = 'predictions_ionic_SMNet_testdata_20000_0.0015.csv'
model_save_path = "SMNet_regression_ionic_20000_0.0015.pth"
df_featureTrain = pd.read_csv('/workspace/inputionic_train.csv', header=None)
df_featureTest = pd.read_csv('/workspace/inputionic_test.csv', header=None)
df_xTrain = pd.read_csv('/workspace/x_ionictrain_20000_0.0015.csv', header=None)
df_yTrain = pd.read_csv('/workspace/y_ionictrain_20000_0.0015.csv', header=None)
df_zTrain = pd.read_csv('/workspace/z_ionictrain_20000_0.0015.csv', header=None)
df_xTest = pd.read_csv('/workspace/x_ionictest_20000_0.0015.csv', header=None)
df_yTest = pd.read_csv('/workspace/y_ionictest_20000_0.0015.csv', header=None)
df_zTest = pd.read_csv('/workspace/z_ionictest_20000_0.0015.csv', header=None)
X_train = np.stack((df_xTrain, df_yTrain, df_zTrain), axis=-1)
X_featureTrain = np.array(df_featureTrain)
X_test = np.stack((df_xTest, df_yTest, df_zTest), axis=-1)
X_featureTest = np.array(df_featureTest)
Z = np.array(df_zTrain)

X_train = normalize_point_clouds(X_train)
X_test = normalize_point_clouds(X_test)

train_input_data = X_train
train_output_data = torch.from_numpy(X_featureTrain).float()
val_input_data = X_test
val_output_data = torch.from_numpy(X_featureTest).float()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transfer to tensor
train_samples = []
val_samples = []
for input_points, output_values in zip(train_input_data, train_output_data):
    pos = torch.tensor(input_points, dtype=torch.float).to(device)
    x = torch.ones((pos.size(0), 1)).to(device)
    train_samples.append(Data(pos=pos, x=x))

for input_points, output_values in zip(val_input_data, val_output_data):
    pos = torch.tensor(input_points, dtype=torch.float).to(device)
    x = torch.ones((pos.size(0), 1)).to(device)
    val_samples.append(Data(pos=pos, x=x))

# Train the model
input_nc = 0
num_classes = 6
output_dim = 216
batch_size = 8
num_points = 7392
model = SMNetRegressor(input_nc, num_classes, 1024, output_dim)
mse_loss = MSELoss()
mae_loss = L1Loss()
model = model.to(device)
train_output_data = train_output_data.to(device)
val_output_data = val_output_data.to(device)
total_params = sum(p.numel() for p in model.parameters())
print(model)
print(f'Total number of parameters: {total_params}')
n_data = 20000 # Number of training trials
n_test_data = 100 # Number of testing trials
n_batches = (n_data + batch_size - 1) // batch_size
n_test_batches = (n_test_data + batch_size - 1) // batch_size - 1 # For batch size 8, there is a -1. For batch size 4, there is no -1.
train_losses = []
train_maes = []
val_losses = []
val_maes = []
# Choose a proper optimizer here
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
# optimizer = optim.Adam(model.parameters(), lr=0.001)
best_r2 = -np.inf
best_model = None
start_time = time.time()


for epoch in range(num_epochs):
    epoch_starttime = time.time()
    model.train()
    train_loss = 0
    train_mae = 0
    train_preds, train_labels = [], []
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i+1) * batch_size, n_data)
        current_batch_data = train_samples[start_idx:end_idx]
        batch = Batch.from_data_list(current_batch_data)
        batch = batch.to(device)
        optimizer.zero_grad()
        output = model.forward(batch, batch_size, num_points, num_classes)
        current_labels = train_output_data[start_idx:end_idx]

        loss = mse_loss(output, current_labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_mae += mae_loss(output, current_labels).item()
        train_preds.append(output.detach().cpu().numpy())
        train_labels.append(current_labels.cpu().numpy())

    train_loss /= n_batches
    train_mae /= n_batches
    train_preds = np.concatenate(train_preds, axis=0)
    train_labels = np.concatenate(train_labels, axis=0)
    train_r2 = r2_score(train_labels, train_preds)

    model.eval()
    test_loss = 0
    test_mae = 0
    test_preds, test_labels = [], []
    outputs = []
    for i in range(n_test_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_test_data)
        current_batch_test_data = val_samples[start_idx:end_idx]
        batch_test = Batch.from_data_list(current_batch_test_data)
        batch_test = batch_test.to(device)
        output = model.forward(batch_test, batch_size, num_points, num_classes)
        current_test_labels = val_output_data[start_idx:end_idx]
        loss = mse_loss(output, current_test_labels)
        outputs.extend(output.tolist())

        test_loss += loss.item()
        test_mae += mae_loss(output, current_test_labels).item()
        test_preds.append(output.detach().cpu().numpy())
        test_labels.append(current_test_labels.cpu().numpy())

    test_loss /= n_test_batches
    test_mae /= n_test_batches
    test_preds = np.concatenate(test_preds, axis=0)
    test_labels = np.concatenate(test_labels, axis=0)
    test_r2 = r2_score(test_labels, test_preds)
    if test_r2 > best_r2:
        best_r2 = test_r2
        best_model = model.state_dict()
        column_names = [f"Value_{i}" for i in range(output_dim)]
        df_output = pd.DataFrame(outputs, columns=column_names)
        df_output.to_csv(csv_name, index=False, header=False)
        torch.save(best_model, model_save_path)
        print('model and testdata saved')
    print(f"Epoch: {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train MAE: {train_mae:.4f}, Train R2: {train_r2:.4f}, Val Loss: {test_loss:.4f}, Val MAE: {test_mae:.4f}, Test R2: {test_r2:.4f}")
    epoch_endtime = time.time()
    epoch_time = epoch_endtime - epoch_starttime
    print(epoch_time / 60)
    print(best_r2)

    model_test = SMNetRegressor(input_nc, num_classes, 1024, output_dim)
    model_test = model_test.to(device)
    model_test.load_state_dict(torch.load(model_save_path))
    model_test.eval()
    test_loss = 0
    test_mae = 0
    test_r2 = 0
    test_preds, test_labels = [], []
    outputs = []
    for i in range(n_test_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_test_data)
        current_batch_test_data = val_samples[start_idx:end_idx]
        batch_test = Batch.from_data_list(current_batch_test_data)
        batch_test = batch_test.to(device)
        output = model_test.forward(batch_test, batch_size, num_points, num_classes)
        outputs.extend(output.tolist())
        current_test_labels = val_output_data[start_idx:end_idx]
        loss = mse_loss(output, current_test_labels)

        test_loss += loss.item()
        test_mae += mae_loss(output, current_test_labels).item()
        test_preds.append(output.detach().cpu().numpy())
        test_labels.append(current_test_labels.cpu().numpy())

    test_loss /= n_test_batches
    test_mae /= n_test_batches
    test_preds = np.concatenate(test_preds, axis=0)
    test_labels = np.concatenate(test_labels, axis=0)
    test_r2 = r2_score(test_labels, test_preds)

    print('R2 score of independent test: {:.2f}'.format(test_r2))
print(best_r2)
end_time = time.time()
train_time = end_time - start_time
print('Training time: {:.2f} minutes'.format(train_time / 60))
print("Training complete.")
print(f"Model saved to {model_save_path}")
