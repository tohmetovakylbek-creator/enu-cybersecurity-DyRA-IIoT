import torch
import torch.nn as nn
import torch.optim as optim


# 1. Блок с остаточной связью (Residual Block) - основа архитектуры TiDE
class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim, dropout=0.2):
        super(ResidualBlock, self).__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # Сохраняем вход для остаточной связи (Skip connection)
        identity = x
        out = self.linear(x)
        out = self.relu(out)
        out = self.dropout(out)
        # Складываем выход со входом и нормализуем
        return self.norm(identity + out)


# 2. Основная архитектура классификатора на базе TiDE
class TiDEAnomalyDetector(nn.Module):
    def __init__(self, seq_len=50, num_features=58, hidden_dim=256, num_layers=2, dropout=0.2):
        super(TiDEAnomalyDetector, self).__init__()

        # Сглаживаем временное окно в один вектор (Flattening)
        self.flatten = nn.Flatten()

        # Проекция входных данных в скрытое пространство
        input_dim = seq_len * num_features
        self.feature_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )

        # Плотный энкодер (Dense Encoder) из нескольких Residual блоков
        self.encoder = nn.Sequential(
            *[ResidualBlock(hidden_dim, dropout) for _ in range(num_layers)]
        )

        # Финальный классификатор (выдает вероятность от 0 до 1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Важно: Sigmoid дает вероятность P(A)
        )

    def forward(self, x):
        # x имеет размерность: [Batch, Seq_Len, Features]
        x_flat = self.flatten(x)  # [Batch, Seq_Len * Features]
        projected = self.feature_projection(x_flat)  # [Batch, hidden_dim]
        encoded = self.encoder(projected)  # [Batch, hidden_dim]
        prob = self.classifier(encoded)  # [Batch, 1]

        return prob.squeeze(-1)  # Возвращаем вектор вероятностей [Batch]


# 3. Функция обучения одной эпохи
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        # Прямой проход (Forward pass)
        optimizer.zero_grad()
        predictions = model(X_batch)

        # Вычисление ошибки
        loss = criterion(predictions, y_batch)

        # Обратное распространение (Backward pass)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Подсчет точности (порог 0.5)
        predicted_classes = (predictions > 0.5).float()
        correct += (predicted_classes == y_batch).sum().item()
        total += y_batch.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy


# Блок для быстрого тестирования архитектуры
if __name__ == "__main__":
    # Создаем фиктивные данные той же размерности, что выдал скрипт 1_dataset_loader.py
    batch_size = 128
    seq_len = 50
    num_features = 58

    dummy_X = torch.rand(batch_size, seq_len, num_features)

    # Инициализация модели
    model = TiDEAnomalyDetector(seq_len=seq_len, num_features=num_features)

    print("Архитектура модели:")
    print(model)

    # Тестовый прогон
    output_probs = model(dummy_X)
    print(f"\nРазмерность выхода модели: {output_probs.shape}")
    print(f"Пример предсказанной вероятности P(A): {output_probs[0].item():.4f}")