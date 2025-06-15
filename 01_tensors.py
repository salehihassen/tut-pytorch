import torch
import numpy as np


def tensor_init() -> None:
    data = [[1, 2], [3, 4]]
    x_data = torch.tensor(data)
    print(x_data)


def from_numpy_array() -> None:
    data = [[1, 2], [3, 4]]
    np_array = np.array(data)
    x_tensor = torch.from_numpy(np_array)
    print(x_tensor)


def from_another_tensor() -> None:
    data = [[1, 2], [3, 4]]
    x_data = torch.tensor(data)
    x_ones = torch.ones_like(x_data)  # retains the properties of x_data
    print(f"Ones Tensor: \n {x_ones} \n")
    x_rand = torch.rand_like(
        x_data, dtype=torch.float
    )  # overrides the datatype of x_data
    print(f"Random Tensor: \n {x_rand} \n")


def rand_tensor_from_shape() -> None:
    shape = (4, 3, 2)
    rand_tensor = torch.rand(shape)
    # ones_tensor = torch.ones(shape)
    # zeros_tensor = torch.zeros(shape)
    print(f"Random Tensor: \n {rand_tensor} \n")
    # print(f"Ones Tensor: \n {ones_tensor} \n")
    # print(f"Zeros Tensor: \n {zeros_tensor}")


def tensor_attributes() -> None:
    tensor = torch.rand(3, 4)
    print(f"Shape of tensor: {tensor.shape}")
    print(f"Datatype of tensor: {tensor.dtype}")
    print(f"Device tensor is stored on: {tensor.device}")


def numpy_slicing_on_tensor() -> None:
    tensor = torch.ones(4, 4)
    tensor[:, 1] = 0
    print(tensor)


def join_tensors() -> None:
    tensor = torch.ones(4, 4)
    tensor[:, 1] = 0
    t1 = torch.cat([tensor, tensor, tensor], dim=1)
    print(t1)


def element_multiply_tensors() -> None:
    tensor = torch.ones(4, 4)
    tensor[:, 1] = 0
    print(f"tensor.mul(tensor) \n {tensor.mul(2 * tensor)} \n")
    # Alternative syntax:
    # print(f"tensor * tensor \n {tensor * tensor}")


def matrix_multiply_tensors() -> None:
    tensor = torch.ones(4, 4)
    tensor[:, 1] = 0
    print(f"tensor.matmul(tensor.T) \n {tensor.matmul(tensor.T)} \n")
    # Alternative syntax:
    # print(f"tensor @ tensor.T \n {tensor @ tensor.T}")


def inplace_op() -> None:
    tensor = torch.ones(4, 4)
    tensor[:, 1] = 0
    print(tensor, "\n")
    tensor.add_(5)
    print(tensor)


if __name__ == "__main__":
    # tensor_init()
    # from_numpy_array()
    # from_another_tensor()
    # rand_tensor_from_shape()
    # tensor_attributes()
    # numpy_slicing_on_tensor()
    # join_tensors()
    # element_multiply_tensors()
    # matrix_multiply_tensors()
    inplace_op()
