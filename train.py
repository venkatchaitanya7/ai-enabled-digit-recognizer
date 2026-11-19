import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from model import DigitCNN


def train_and_export():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Training High-Accuracy Model on: {device}")

    # Heavy data augmentation for real-world photo resilience
    train_transform = transforms.Compose([
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=train_transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=test_transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

    model = DigitCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    epochs = 10
    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()

        # Validation
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                preds = model(data).argmax(dim=1)
                correct += preds.eq(target).sum().item()

        accuracy = 100. * correct / len(test_dataset)
        print(f"Epoch [{epoch:02d}/{epochs:02d}] - Loss: {running_loss/len(train_loader):.4f} - Test Accuracy: {accuracy:.2f}%")
        best_acc = max(best_acc, accuracy)

    print(f"\n✅ Training Complete! Peak Accuracy: {best_acc:.2f}%")

    # 1. Save PyTorch weights
    torch.save(model.state_dict(), 'digit_model.pth')
    print("📦 Saved 'digit_model.pth'")

    # 2. Export to lightweight ONNX format
    model.eval()
    dummy_input = torch.randn(1, 1, 28, 28, device=device)
    torch.onnx.export(
        model,
        dummy_input,
        "digit_model.onnx",
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
        dynamo=False  # Crucial fix to disable the experimental compiler and export successfully!
    )
    print("📦 Exported 'digit_model.onnx' (Ultra-lightweight for deployment)!")


if __name__ == '__main__':
    train_and_export()