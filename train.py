import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from model import DigitCNN

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Training on device: {device}")

    # Data Augmentation: prepares the AI for real-world messy handwriting
    train_transform = transforms.Compose([
        transforms.RandomRotation(degrees=12),
        transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.95, 1.05)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load datasets
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=train_transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=test_transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

    model = DigitCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.7)

    epochs = 8  # Optimal balance between high accuracy and preventing overfitting
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()

        # Evaluate test accuracy per epoch
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                preds = model(data).argmax(dim=1)
                correct += preds.eq(target).sum().item()

        accuracy = 100. * correct / len(test_dataset)
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch:02d}/{epochs:02d} | Loss: {avg_loss:.4f} | Test Accuracy: {accuracy:.2f}%")

    # Save the upgraded model
    torch.save(model.state_dict(), 'digit_model.pth')
    print("✅ High-accuracy model saved to 'digit_model.pth'")

if __name__ == '__main__':
    train()
