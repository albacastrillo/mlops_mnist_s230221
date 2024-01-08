import torch


def mnist():
    """Return train and test dataloaders for MNIST."""
    train_data, train_labels = [], []

    # Load the training data
    for i in range(5):  # There are 5 sets of training data
        train_data.append(torch.load(f"train_images_{i}.pt"))
        train_labels.append(torch.load(f"train_target_{i}.pt"))

    train_data = torch.cat(train_data, dim=0)
    train_labels = torch.cat(train_labels, dim=0)

    # Load the test data
    test_data = torch.load("test_images.pt")
    test_labels = torch.load("test_target.pt")

    print(train_data.shape)
    print(train_labels.shape)
    print(test_data.shape)
    print(test_labels.shape)

    train_data = train_data.unsqueeze(1)
    test_data = test_data.unsqueeze(1)


    return (
        torch.utils.data.TensorDataset(train_data, train_labels), 
        torch.utils.data.TensorDataset(test_data, test_labels)
    )
