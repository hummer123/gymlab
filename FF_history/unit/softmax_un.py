import numpy as np

def softmax(x, axis: int = -1) -> np.ndarray:
    """
    带数值稳定的Softmax函数实现.

    Parameters
    ----------
    x : ndarray 输入数组（可以是1D/2D/更高维）
        Input data.
    axis : int, optional
        Axis along which the softmax is computed. By default axis is None,
        and the softmax is computed over the entire array.

    Returns
    -------
    ndarray 概率分布数组（形状与输入一致，每行/每列总和为1）
        An array the same shape as x. The result will sum to 1 along the
        specified axis.
    """
    # Subtract the max for numerical stability
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    sum_e_x = np.sum(e_x, axis=axis, keepdims=True)
    return e_x / sum_e_x


if __name__ == "__main__":
    # Test the softmax function
    test_array = np.array([[2.0, 1.0, 0.1]])
    print("Input Array:")
    print(test_array)
    print("Softmax along last axis:")
    print(softmax(test_array, axis=-1))
    print("Softmax along first axis:")
    print(softmax(test_array, axis=0))