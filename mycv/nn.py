"""
mycv.nn
=======
A small, pure-NumPy neural-network module: Conv2D, pooling, dense layers,
activations, a softmax-cross-entropy loss, SGD, and a Sequential container
for chaining them into a trainable CNN.

This is deliberately minimal ("start small") -- it augments the classical
CV toolkit with a learned option where hand-crafted heuristics (like
`features.classify_object`'s aspect-ratio/circularity shape rule) hit
their limits, not a general deep-learning framework. No autograd: every
layer implements its own analytic forward/backward pass, in the same
from-first-principles spirit as the rest of `mycv`.

Scope and honest limitations
------------------------------
- CPU-only, pure NumPy: no GPU, no fused kernels, no autodiff graph
  optimisation. This is adequate for small images (tens of pixels per
  side) and small networks (a few conv layers), trained on a few
  thousand examples in minutes on a laptop -- NOT for deep networks or
  large datasets. Use PyTorch/TensorFlow for anything beyond that.
- `MaxPool2D`/`AvgPool2D` only support non-overlapping pooling
  (stride == pool_size). This keeps the backward pass exact and simple
  (each input pixel contributes to exactly one pooling window) at the
  cost of not supporting overlapping pooling windows.
- `Conv2D` supports 'valid' or 'same' padding and arbitrary stride.

Layers
------
Conv2D    : multi-channel, multi-filter 2-D convolution (learnable)
MaxPool2D : non-overlapping max pooling
AvgPool2D : non-overlapping average pooling
Flatten   : (N, C, H, W) <-> (N, C*H*W)
Dense     : fully-connected layer (learnable)
ReLU      : elementwise rectifier

Functions
---------
softmax_cross_entropy : combined softmax + cross-entropy loss and gradient
                         (numerically stable via the log-sum-exp trick)

Classes
-------
SGD        : stochastic gradient descent optimiser, with optional momentum
Sequential : chains layers into a model; forward/backward/predict/save/load

Conv2D forward/backward: the "shift-and-add" method
-----------------------------------------------------
A naive im2col-style implementation materialises a full
(N, C_in, kH, kW, H_out, W_out) patch tensor, which for many
channels/filters can be very large. Instead, `Conv2D` loops over the
kH*kW kernel OFFSETS (a small, fixed number -- e.g. 9 for a 3x3 kernel),
and for each offset (i, j) takes a single strided slice of the padded
input:

    patch_ij = padded[:, :, i : i + stride*H_out : stride,
                             j : j + stride*W_out : stride]        # (N, C_in, H_out, W_out)

and accumulates that offset's contribution to the output via one
matrix contraction:

    out += einsum('ncyx,oc->noyx', patch_ij, W[:, :, i, j])

For a FIXED (i, j), the strided slice picks out non-overlapping input
positions (they're spaced `stride` apart), so accumulating the input
gradient back into those same slice positions during backprop is exact
with a simple `+=` -- no scatter-add / col2im machinery needed. This is
mathematically equivalent to the standard im2col convolution, just
reorganised to keep peak memory at O(N * C * H_out * W_out) instead of
O(N * C * kH * kW * H_out * W_out).
"""

import numpy as np


# ---------------------------------------------------------------------------
# Padding helper
# ---------------------------------------------------------------------------

def _pad_hw(x: np.ndarray, pad_h: int, pad_w: int) -> np.ndarray:
    """Zero-pad the last two axes (H, W) of a (N, C, H, W) tensor."""
    if pad_h == 0 and pad_w == 0:
        return x
    return np.pad(x, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)),
                  mode="constant", constant_values=0.0)


def _same_padding(kH: int, kW: int) -> tuple:
    """Symmetric padding that keeps output size == input size for stride=1."""
    return kH // 2, kW // 2


# ---------------------------------------------------------------------------
# Conv2D
# ---------------------------------------------------------------------------

class Conv2D:
    """
    Multi-channel 2-D convolution layer (cross-correlation, the standard
    CNN convention -- no kernel flip, unlike `filters.convolve2d`).

    Weight shape: (out_channels, in_channels, kH, kW)
    Bias shape:   (out_channels,)

    Input:  (N, C_in, H, W)
    Output: (N, C_out, H_out, W_out)

    Parameters
    ----------
    in_channels, out_channels : int
    kernel_size : int or (kH, kW)
    stride      : int (default 1)
    padding     : 'same' (default) or 'valid'
    seed        : int, optional -- for reproducible He-initialised weights
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding="same", seed=None):
        kH, kW = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kH, self.kW = int(kH), int(kW)
        self.stride = int(stride)
        self.padding = padding

        rng = np.random.default_rng(seed)
        # He initialisation: keeps activation variance stable through ReLU layers.
        fan_in = self.in_channels * self.kH * self.kW
        scale = np.sqrt(2.0 / fan_in)
        self.W = rng.normal(0.0, scale, size=(self.out_channels, self.in_channels, self.kH, self.kW))
        self.b = np.zeros(self.out_channels, dtype=np.float64)

        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

        self._cache = None

    def _pad_amount(self):
        if self.padding == "same":
            return _same_padding(self.kH, self.kW)
        elif self.padding == "valid":
            return 0, 0
        raise ValueError("padding must be 'same' or 'valid'.")

    def output_shape(self, H, W):
        pad_h, pad_w = self._pad_amount()
        H_out = (H + 2 * pad_h - self.kH) // self.stride + 1
        W_out = (W + 2 * pad_w - self.kW) // self.stride + 1
        return H_out, W_out

    def forward(self, x: np.ndarray) -> np.ndarray:
        N, C, H, W = x.shape
        if C != self.in_channels:
            raise ValueError(f"Conv2D expected {self.in_channels} input channels, got {C}.")

        pad_h, pad_w = self._pad_amount()
        padded = _pad_hw(x, pad_h, pad_w)
        H_out, W_out = self.output_shape(H, W)

        out = np.zeros((N, self.out_channels, H_out, W_out), dtype=np.float64)

        for i in range(self.kH):
            for j in range(self.kW):
                patch = padded[:, :, i:i + self.stride * H_out:self.stride,
                                     j:j + self.stride * W_out:self.stride]
                # (N, C_in, H_out, W_out) x (C_out, C_in) -> (N, C_out, H_out, W_out)
                out += np.einsum("ncyx,oc->noyx", patch, self.W[:, :, i, j])

        out += self.b[np.newaxis, :, np.newaxis, np.newaxis]

        self._cache = (padded.shape, x.shape, H_out, W_out)
        self._last_padded = padded
        return out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        padded_shape, x_shape, H_out, W_out = self._cache
        padded = self._last_padded
        pad_h, pad_w = self._pad_amount()

        self.dW.fill(0.0)
        self.db[:] = grad_output.sum(axis=(0, 2, 3))

        grad_padded = np.zeros(padded_shape, dtype=np.float64)

        for i in range(self.kH):
            for j in range(self.kW):
                patch = padded[:, :, i:i + self.stride * H_out:self.stride,
                                     j:j + self.stride * W_out:self.stride]
                # Weight gradient: sum over N, H_out, W_out of patch * grad_output
                self.dW[:, :, i, j] = np.einsum("noyx,ncyx->oc", grad_output, patch)

                # Input gradient: scatter grad_output back through W (exact,
                # since this strided slice touches disjoint padded positions
                # for a fixed (i, j) -- see module docstring).
                grad_padded[:, :, i:i + self.stride * H_out:self.stride,
                                  j:j + self.stride * W_out:self.stride] += \
                    np.einsum("noyx,oc->ncyx", grad_output, self.W[:, :, i, j])

        if pad_h == 0 and pad_w == 0:
            grad_input = grad_padded
        else:
            grad_input = grad_padded[:, :, pad_h:pad_h + x_shape[2], pad_w:pad_w + x_shape[3]]

        return grad_input

    def params_and_grads(self):
        return [(self.W, self.dW), (self.b, self.db)]


# ---------------------------------------------------------------------------
# Pooling
# ---------------------------------------------------------------------------

class MaxPool2D:
    """
    Non-overlapping 2-D max pooling (stride == pool_size).

    Input:  (N, C, H, W)  -- H, W must be divisible by pool_size
    Output: (N, C, H/pool_size, W/pool_size)
    """

    def __init__(self, pool_size=2):
        self.k = int(pool_size)
        self._cache = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        N, C, H, W = x.shape
        k = self.k
        if H % k != 0 or W % k != 0:
            raise ValueError(
                f"MaxPool2D: H={H}, W={W} must both be divisible by pool_size={k}."
            )
        H_out, W_out = H // k, W // k

        # Reshape into non-overlapping k x k blocks: (N, C, H_out, k, W_out, k)
        reshaped = x.reshape(N, C, H_out, k, W_out, k)
        out = reshaped.max(axis=(3, 5))

        # Build an argmax mask for the backward pass (which element in each
        # window was the max -- ties broken by splitting gradient equally,
        # see `backward`).
        out_broadcast = out[:, :, :, np.newaxis, :, np.newaxis]
        is_max = (reshaped == out_broadcast)
        self._cache = (x.shape, is_max)
        return out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        x_shape, is_max = self._cache
        N, C, H, W = x_shape
        k = self.k
        H_out, W_out = H // k, W // k

        grad_broadcast = grad_output[:, :, :, np.newaxis, :, np.newaxis]

        # Normalise by the count of tied maxima per window so gradient sums
        # to exactly grad_output (handles the rare exact-tie case correctly;
        # for the common no-tie case this divides by 1, i.e. a no-op).
        tie_count = is_max.sum(axis=(3, 5), keepdims=True)
        grad_windows = np.where(is_max, grad_broadcast / tie_count, 0.0)

        return grad_windows.reshape(N, C, H, W)

    def params_and_grads(self):
        return []


class AvgPool2D:
    """
    Non-overlapping 2-D average pooling (stride == pool_size).
    """

    def __init__(self, pool_size=2):
        self.k = int(pool_size)
        self._x_shape = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        N, C, H, W = x.shape
        k = self.k
        if H % k != 0 or W % k != 0:
            raise ValueError(
                f"AvgPool2D: H={H}, W={W} must both be divisible by pool_size={k}."
            )
        self._x_shape = x.shape
        H_out, W_out = H // k, W // k
        reshaped = x.reshape(N, C, H_out, k, W_out, k)
        return reshaped.mean(axis=(3, 5))

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        N, C, H, W = self._x_shape
        k = self.k
        H_out, W_out = H // k, W // k
        # Distribute gradient equally to every element in its window.
        grad = grad_output[:, :, :, np.newaxis, :, np.newaxis] / (k * k)
        grad = np.broadcast_to(grad, (N, C, H_out, k, W_out, k))
        return grad.reshape(N, C, H, W)

    def params_and_grads(self):
        return []


# ---------------------------------------------------------------------------
# Flatten / Dense / activations
# ---------------------------------------------------------------------------

class Flatten:
    """(N, C, H, W) -> (N, C*H*W), and back on the backward pass."""

    def __init__(self):
        self._shape = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        return grad_output.reshape(self._shape)

    def params_and_grads(self):
        return []


class Dense:
    """
    Fully-connected layer: y = x @ W.T + b

    Weight shape: (out_features, in_features)
    Bias shape:   (out_features,)
    """

    def __init__(self, in_features, out_features, seed=None):
        rng = np.random.default_rng(seed)
        scale = np.sqrt(2.0 / in_features)
        self.W = rng.normal(0.0, scale, size=(out_features, in_features))
        self.b = np.zeros(out_features, dtype=np.float64)

        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

        self._x = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        return x @ self.W.T + self.b

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        self.dW[:] = grad_output.T @ self._x
        self.db[:] = grad_output.sum(axis=0)
        return grad_output @ self.W

    def params_and_grads(self):
        return [(self.W, self.dW), (self.b, self.db)]


class ReLU:
    """Elementwise rectified linear unit: max(0, x)."""

    def __init__(self):
        self._mask = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._mask = x > 0
        return x * self._mask

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        return grad_output * self._mask

    def params_and_grads(self):
        return []


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def softmax_cross_entropy(logits: np.ndarray, labels: np.ndarray) -> tuple:
    """
    Combined softmax + cross-entropy loss, with its gradient wrt `logits`.

    Computing these together (rather than a separate Softmax layer feeding
    a cross-entropy loss) both is numerically stable (log-sum-exp trick,
    avoiding overflow in exp()) and gives a very simple gradient:

        d(loss)/d(logits) = softmax(logits) - one_hot(labels)

    Parameters
    ----------
    logits : np.ndarray  shape (N, num_classes) -- raw (pre-softmax) scores
    labels : np.ndarray  shape (N,) -- integer class indices in [0, num_classes)

    Returns
    -------
    (loss, grad) : (float, np.ndarray shape (N, num_classes))
        loss is the mean cross-entropy over the batch.
    """
    N = logits.shape[0]

    # log-sum-exp trick: subtract the row max before exponentiating.
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)

    log_probs = shifted - np.log(exp_scores.sum(axis=1, keepdims=True))
    loss = -log_probs[np.arange(N), labels].mean()

    grad = probs.copy()
    grad[np.arange(N), labels] -= 1.0
    grad /= N

    return float(loss), grad


# ---------------------------------------------------------------------------
# Optimiser
# ---------------------------------------------------------------------------

class SGD:
    """
    Stochastic gradient descent, with optional momentum.

        v = momentum * v - lr * grad
        param += v

    Parameters
    ----------
    lr       : float  learning rate
    momentum : float  in [0, 1); 0 disables momentum (plain SGD)
    """

    def __init__(self, lr=0.01, momentum=0.0):
        self.lr = float(lr)
        self.momentum = float(momentum)
        self._velocity = {}

    def step(self, params_and_grads: list) -> None:
        for i, (param, grad) in enumerate(params_and_grads):
            if self.momentum > 0:
                v = self._velocity.get(i, np.zeros_like(param))
                v = self.momentum * v - self.lr * grad
                self._velocity[i] = v
                param += v
            else:
                param -= self.lr * grad


# ---------------------------------------------------------------------------
# Sequential container
# ---------------------------------------------------------------------------

class Sequential:
    """
    Chains a list of layers into a model.

    Each layer must implement `forward(x)`, `backward(grad_output)`, and
    `params_and_grads()` (returning a list of (param, grad) array pairs,
    empty for layers with no learnable parameters).

    Parameters
    ----------
    layers : list of layer instances, in forward-pass order
    """

    def __init__(self, layers: list):
        self.layers = layers

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)
        return grad_output

    def params_and_grads(self) -> list:
        out = []
        for layer in self.layers:
            out.extend(layer.params_and_grads())
        return out

    def train_step(self, x: np.ndarray, labels: np.ndarray, optimizer: "SGD") -> float:
        """
        One forward + softmax-cross-entropy loss + backward + optimiser
        step. Returns the batch loss.
        """
        logits = self.forward(x)
        loss, grad = softmax_cross_entropy(logits, labels)
        self.backward(grad)
        optimizer.step(self.params_and_grads())
        return loss

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Return predicted class probabilities (N, num_classes), softmax of forward()."""
        logits = self.forward(x)
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp_scores = np.exp(shifted)
        return exp_scores / exp_scores.sum(axis=1, keepdims=True)

    def save_weights(self, path: str) -> None:
        """Save every learnable parameter to a single .npz file, in layer order."""
        arrays = {}
        idx = 0
        for layer in self.layers:
            for param, _ in layer.params_and_grads():
                arrays[f"param_{idx}"] = param
                idx += 1
        np.savez(path, **arrays)

    def load_weights(self, path: str) -> None:
        """Load parameters saved by `save_weights` into this model, in place."""
        data = np.load(path)
        idx = 0
        for layer in self.layers:
            for param, _ in layer.params_and_grads():
                key = f"param_{idx}"
                if key not in data:
                    raise ValueError(
                        f"Weight file is missing '{key}' -- does this model's "
                        f"architecture match the one the file was saved from?"
                    )
                saved = data[key]
                if saved.shape != param.shape:
                    raise ValueError(
                        f"Shape mismatch loading '{key}': model expects "
                        f"{param.shape}, file has {saved.shape}."
                    )
                param[...] = saved
                idx += 1
