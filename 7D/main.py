import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import time

# 裝置設定：自動使用 GPU（若可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def prepare_data(file_path):
    data = pd.read_excel(file_path)
    X = data.iloc[:, :-1].values.astype(np.float32)
    y = data.iloc[:, -1:].values.astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1
    )

    train_data = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    test_data  = TensorDataset(torch.tensor(X_test),  torch.tensor(y_test))
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader  = DataLoader(test_data,  batch_size=32, shuffle=False)

    return train_loader, test_loader, X_train.shape[1]

# 兩層全連接網路 (線性輸出)
class TwoLayerNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TwoLayerNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        return self.fc2(h)
    
class TwoLayerNet_Dropout(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_prob):
        super(TwoLayerNet_Dropout, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        h = self.dropout(h)
        return self.fc2(h)
    
def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, loss_threshold, max_t, check_interval):
    train_losses = []
    test_losses  = []
    t = 0
    result = False
    while True:
        if t % check_interval == 0:
            model.eval()
            total_loss = 0.0
            with torch.no_grad():
                for x_batch, y_batch in test_loader:
                    x_batch = x_batch.to(device)
                    y_batch = y_batch.to(device)
                    output = model(x_batch)
                    loss = criterion(output, y_batch)
                    total_loss += loss.item()
            avg_loss = total_loss / len(test_loader)
            test_losses.append((t, avg_loss))
            print(f"t={t}: 測試損失={avg_loss:.4f}")

            if avg_loss < 0.25:
                print("An acceptable 2LNN")
                result = True
                break

            if avg_loss < loss_threshold:
                print(f"t={t}: 測試損失達標，訓練結束")
                break

        if t >= max_t:
            print(f"t={t}: 達最大迭代次數，訓練結束")
            break

        model.train()
        total_train = 0.0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            output = model(x_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            total_train += loss.item()

        avg_train = total_train / len(train_loader)
        train_losses.append((t, avg_train))

        scheduler.step()
        t += 1

    return train_losses, test_losses, result

def selecting_LTS(model, data_loader, epsilon):
    """
    LTS原則：列出所有 residual > epsilon^2 的樣本，並挑選 residual 最大的樣本。
    傳回：最大 residual 的 index、x_k、y_k
    """
    model.eval()
    residuals = []
    inputs = []
    targets = []
    all_indices = []

    with torch.no_grad():
        index_offset = 0
        for x_batch, y_batch in data_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(x_batch)
            loss = (outputs - y_batch) ** 2
            batch_residuals = loss.sum(dim=1).cpu().numpy()

            residuals.extend(batch_residuals)
            inputs.extend(x_batch.cpu().numpy())
            targets.extend(y_batch.cpu().numpy())
            all_indices.extend(range(index_offset, index_offset + len(x_batch)))
            index_offset += len(x_batch)

    residuals = np.array(residuals)
    all_indices = np.array(all_indices)
    inputs = np.array(inputs)
    targets = np.array(targets)

    mask = residuals > epsilon**2
    bad_indices = all_indices[mask]
    bad_residuals = residuals[mask]
    bad_inputs = inputs[mask]
    bad_targets = targets[mask]

    if len(bad_indices) == 0:
        print(f"所有樣本 residual 都小於 ε² = {epsilon ** 2:.4f}")
        return None, None, None

    print(f"未通過門檻的樣本數：{len(bad_indices)}（ε² = {epsilon ** 2:.4f}）")
    # for i, r in zip(bad_indices, bad_residuals):
    #     print(f"Index: {i}, Residual: {r:.4f}")

    # 找出最大 residual 的樣本
    max_idx = np.argmax(bad_residuals)
    selected_index = bad_indices[max_idx]
    x_k = bad_inputs[max_idx]
    y_k = bad_targets[max_idx]

    print(f"挑選 residual 最大的樣本：Index = {selected_index}, Residual = {bad_residuals[max_idx]:.4f}")
    return selected_index, x_k, y_k

def add_random_node(model):
    """
    在 TwoLayerNet 隱藏層中新增 1 個隨機初始化節點
    """
    with torch.no_grad():
        # 備份舊參數
        old_w1 = model.fc1.weight.data.clone()
        old_b1 = model.fc1.bias.data.clone()
        old_w2 = model.fc2.weight.data.clone()
        old_b2 = model.fc2.bias.data.clone()

        H, D = old_w1.shape
        O = old_w2.shape[0]

        # 建立新層
        model.fc1 = nn.Linear(D, H+1).to(old_w1.device)
        model.fc2 = nn.Linear(H+1, O).to(old_w1.device)

        # 原權重賦值
        model.fc1.weight.data[:H] = old_w1
        model.fc1.bias.data[:H] = old_b1
        model.fc2.weight.data[:, :H] = old_w2
        model.fc2.bias.data = old_b2.clone()

        # 新增節點權重隨機初始化
        nn.init.xavier_uniform_(model.fc1.weight.data[H:].unsqueeze(0))
        model.fc1.bias.data[H] = 0.0
        # 輸出層到新節點的權重隨機
        nn.init.xavier_uniform_(model.fc2.weight.data[:, H:].unsqueeze(1))

        print(f"新增 1 個隨機節點，隱藏層維度: {H+1}")

def prune_nodes(model, train_loader, test_loader, criterion):
    """
    逐一嘗試移除隱藏層節點：若移除後 loss 降低，則保留剪枝。
    回傳剪枝後的 model。
    """
    # 先計算原始 loss
    best_loss = evaluate_loss(model, test_loader, criterion)
    print(f"[Prune] 原始測試 loss = {best_loss:.4f}")

    # 把原參數備份成 numpy
    w1 = model.fc1.weight.data.cpu().numpy()
    b1 = model.fc1.bias.data.cpu().numpy()
    w2 = model.fc2.weight.data.cpu().numpy()
    b2 = model.fc2.bias.data.cpu().numpy()

    input_size, H = w1.shape[1], w1.shape[0]
    idx = 0

    # 每次嘗試移除一個節點，H 會逐步減少
    while idx < H:
        # 建立新權重：去掉第 idx 列／欄
        w1_new = np.delete(w1, idx, axis=0)      # (H-1, D)
        b1_new = np.delete(b1, idx, axis=0)      # (H-1,)
        w2_new = np.delete(w2, idx, axis=1)      # (O, H-1)

        # 重建模型
        pruned = TwoLayerNet(input_size, H-1).to(device)
        pruned.fc1.weight.data.copy_(torch.from_numpy(w1_new))
        pruned.fc1.bias.data.copy_(torch.from_numpy(b1_new))
        pruned.fc2.weight.data.copy_(torch.from_numpy(w2_new))
        pruned.fc2.bias.data.copy_(torch.from_numpy(b2))

        # 權重調整
        optimizer = torch.optim.Adam(pruned.parameters(), lr=0.0005)
        pruned.train()
        for i, (x, y) in enumerate(train_loader):
            if i >= 10: break
            optimizer.zero_grad()
            loss = criterion(pruned(x.to(device)), y.to(device))
            loss.backward()
            optimizer.step()

        # 計算 loss
        loss_pruned = evaluate_loss(pruned, test_loader, criterion)
        print(f"[Prune] 嘗試移除節點 {idx} → loss = {loss_pruned:.4f}", end=" ")

        if loss_pruned < best_loss:
            # 接受剪枝
            print("接受剪枝，移除此節點")
            model = pruned
            best_loss = loss_pruned
            # 更新參數備份、H，不往前移動 idx（因為下一個節點已經往前移一位）
            w1, b1, w2, b2 = w1_new, b1_new, w2_new, b2
            H -= 1
        else:
            # 捨棄剪枝
            print("維持原節點")
            idx += 1

    print(f"[Prune] 最終隱藏節點數：{H}")
    return model

def evaluate_loss(model, data_loader, criterion):
    """回傳 model 在 data_loader 上的平均 loss"""
    model.eval()
    total = 0.0
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            total += criterion(model(x), y).item()
    return total / len(data_loader)

def evaluate_model(model, file_path):
    """
    載入驗證資料並計算 MAE、MAPE、RMSE
    """
    model.eval()
    data = pd.read_excel(file_path)
    X_val = data.iloc[:, :-1].values.astype(np.float32)
    y_val = data.iloc[:, -1:].values.astype(np.float32)

    X_tensor = torch.tensor(X_val).to(device)
    y_tensor = torch.tensor(y_val).to(device)

    with torch.no_grad():
        preds = model(X_tensor).cpu().numpy()
        targets = y_tensor.cpu().numpy()

    mae = mean_absolute_error(targets, preds)
    rmse = np.sqrt(mean_squared_error(targets, preds))
    mape = np.mean(np.abs((targets - preds) / (targets + 1e-8))) * 100  # 防止除以0

    print(f"{file_path} 結果：")
    print(f"MAE  = {mae:.4f}")
    print(f"MAPE = {mape:.2f}%")
    print(f"RMSE = {rmse:.4f}")

if __name__ == "__main__":

    print("使用 GPU: " if torch.cuda.is_available() else "使用 CPU")
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))

    train_loader, test_loader, input_size = prepare_data('training_data_7D.xlsx')

    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)

    model = TwoLayerNet(input_size=input_size, hidden_size=4).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.7)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)

    start_train = time.time()

    train_hist, test_hist, result = train_model(
        model, train_loader, test_loader,
        criterion, optimizer, scheduler,
        loss_threshold=0.29, max_t=50, check_interval=5
    )

    evaluate_model(model, 'validation_data_7D.xlsx')
    if result == False:
        for i in range(10):
            print(f"\n第 {i+1} 次 Select")
            selected_idx, x_k, y_k = selecting_LTS(model, train_loader, epsilon=1.5)
            if x_k is None:
                print("所有樣本預測皆在可接受範圍內")
                break

            add_random_node(model)

            print("\n開始節點加入後的權重調整")
            optimizer = optim.Adam(model.parameters(), lr=0.01)
            scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=1.1)
            train_model(
                model, train_loader, test_loader,
                criterion, optimizer, scheduler,
                loss_threshold=0.3, max_t=50, check_interval=5
            )

        print("\nDO_EB_LG")
        # 取得舊模型參數
        input_size = model.fc1.in_features
        hidden_size = model.fc1.out_features

        # 建立新模型並複製參數
        model_dropout = TwoLayerNet_Dropout(input_size, hidden_size, dropout_prob=0.2).to(device)
        model_dropout.fc1.weight.data = model.fc1.weight.data.clone()
        model_dropout.fc1.bias.data   = model.fc1.bias.data.clone()
        model_dropout.fc2.weight.data = model.fc2.weight.data.clone()
        model_dropout.fc2.bias.data   = model.fc2.bias.data.clone()

        optimizer = optim.Adam(model_dropout.parameters(), lr=0.01)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)

        train_model(
            model_dropout, train_loader, test_loader,
            criterion, optimizer, scheduler,
            loss_threshold=0.29, max_t=50, check_interval=5
        )

        print("\nmodel")
        evaluate_model(model, 'validation_data_7D.xlsx')

        print("\nmodel_dropout")
        evaluate_model(model_dropout, 'validation_data_7D.xlsx')

        print("\nNode-pruning_AN")
        model_pruned = prune_nodes(model_dropout, train_loader, test_loader, criterion)

        print("\nmodel_pruned")
        evaluate_model(model_pruned, 'validation_data_7D.xlsx')

        end_train = time.time()
        print("訓練總時間: {:.4f} 秒".format(end_train - start_train))
    