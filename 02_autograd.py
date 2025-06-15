from numpy import absolute
import torch
import math
from torchvision.models import resnet18, ResNet18_Weights


def one_training_iteration() -> None:
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    data = torch.rand(1, 3, 64, 64)
    labels = torch.rand(1, 1000)

    prediction = model(data)  # forward pass

    # loss = ((prediction - labels).abs()).sum()
    loss = (prediction - labels).sum()
    loss.backward()  # backward pass

    print("Labels")
    print(labels)
    optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
    optim.step()  # gradient descent
    print("Model")
    print(model)
    print("Loss before step")
    print(loss)
    new_prediction = model(data)
    print("Loss after step")
    print((new_prediction - labels).sum())


def differentiation_in_autograd() -> None:
    a = torch.tensor([2.0, 3.0], requires_grad=True)
    b = torch.tensor([6.0, 4.0], requires_grad=True)
    Q = 3 * a**3 - b**2
    print("matrix multiplying")
    print(Q)


def aggregate_q() -> None:
    a = torch.tensor([2.0, 3.0], requires_grad=True)
    b = torch.tensor([6.0, 4.0], requires_grad=True)
    Q = 3 * a**3 - b**2
    external_grad = torch.tensor([1.0, 1.0])
    Q.backward(gradient=external_grad)
    # check if collected gradients are correct
    print(9 * a**2 == a.grad)
    print(-2 * b == b.grad)


def autograd_tracking() -> None:
    x = torch.rand(5, 5)
    y = torch.rand(5, 5)
    z = torch.rand((5, 5), requires_grad=True)
    a = x + y
    print(f"Does `a` require gradients?: {a.requires_grad}")
    b = x + z
    print(f"Does `b` require gradients?: {b.requires_grad}")


def freeze_some_parameters_example():
    # Freeze all the parameters in the network
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    print(model.layer1)
    # print(model.layer2)
    print(model.fc)
    model.fc = torch.nn.Linear(512, 10)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
    optimizer.step()
    print(model.fc)


def compute_minimum_1d_sgd() -> float:
    """Example of using Stochastic Gradient Descent to find the minimum of a one dimensional quadratic equation."""
    # Initialize x with a starting value (e.g., 0.0)
    x = torch.tensor(0.0, requires_grad=True)

    # Define optimizer (SGD) for the parameter x
    optimizer = torch.optim.SGD([x], lr=0.1, momentum=0.7)

    # Training loop
    for epoch in range(100):
        # Forward pass: compute loss
        y = (x + 50) * (x - 30)

        # Backward pass: compute gradients
        optimizer.zero_grad()
        y.backward()

        # Update parameters
        optimizer.step()

        # Optional: print progress
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: x = {x.item():.4f}, y = {y.item():.4f}")

    # Theoretical minimum at x = -10 (vertex of parabola)
    print(f"\nFinal result: x = {x.item():.4f} (Theoretical minimum: x = -10)")
    return x.item()


if __name__ == "__main__":
    # one_training_iteration()
    # differentiation_in_autograd()
    # aggregate_q()
    # autograd_tracking()
    # freeze_some_parameters_example()
    compute_minimum_1d_sgd()
