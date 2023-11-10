import open3d as o3d
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import MSELoss, L1Loss
from torch_points3d.applications.pointnet2 import PointNet2
from torch_geometric.data import Batch, Data, Dataset, DataLoader
import time

def normalize_point_clouds(point_clouds: np.ndarray) -> np.ndarray:
    centroids = np.mean(point_clouds, axis=1)
    point_clouds_centered = point_clouds-centroids[:, np.newaxis, :]
    max_norms = np.max(np.linalg.norm(point_clouds_centered, axis=2), axis=1)
    point_clouds_normalized = point_clouds_centered/max_norms[:, np.newaxis, np.newaxis]
    return point_clouds_normalized

ply_name = "**.ply"
pth_name = '**.pth'
csv_name = '**.csv'
num_pointcloud = 7428 # Pneumatic Model: 5432; Thermal Model: 5048; Ionic Model: 7428
output_dim = 216 # Pneumatic Model: 152; Thermal Model: 152; Ionic Model: 216

cloud = o3d.io.read_point_cloud(ply_name)

points_array = np.asarray(cloud.points)
sorted_indices = np.lexsort((points_array[:, 2], points_array[:, 0], points_array[:, 1]))
sorted_points = points_array[sorted_indices]

print(sorted_points.shape)
num_dataset = 1

reshaped_array = sorted_points.reshape(1, num_pointcloud, 3)
val_input_data = normalize_point_clouds(reshaped_array)

train_samples = []
val_samples = []
class PointNetRegressor(nn.Module):
    def __init__(self, input_nc, num_classes, output_dim):
        super(PointNetRegressor, self).__init__()
        self.pointnet2 = PointNet2(architecture="unet", input_nc=input_nc, num_layers=4, output_nc=num_classes)
        self.regressor = nn.Sequential(
            nn.Linear(num_classes, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
        )

    def forward(self, tdata):
        pointnet2_output = self.pointnet2(tdata)
        x_reshaped = pointnet2_output.x.permute(0, 2, 1)
        pooled_output = x_reshaped.mean(dim=1)
        x = self.regressor(pooled_output)
        return x

model = PointNetRegressor(0, 1024, output_dim)
state_dict = torch.load(pth_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mse_loss = MSELoss()
mae_loss = L1Loss()

for input_points in val_input_data:
    pos = torch.tensor(input_points, dtype=torch.float).unsqueeze(0).to(device)
    val_samples.append(Data(pos=pos))

model = model.to(device)
model.load_state_dict(state_dict)
model.eval()
outputs = []
start_time = time.time()
with torch.no_grad():
    for i in range(num_dataset):
        current_batch_test_data = val_samples[i:i+1]
        batch_test = Batch.from_data_list(current_batch_test_data)
        batch_test = batch_test.to(device)
        output = model.forward(batch_test)
        outputs.extend(output.tolist())
end_time = time.time()
column_names = [f"Value_{i}" for i in range(output_dim)]
df_output = pd.DataFrame(outputs, columns=column_names)
df_output.to_csv(csv_name, index=False, header=False)
executing_time = end_time - start_time
print(executing_time)
print("Prediction complete.")

