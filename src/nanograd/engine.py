"""Nano Grad Engine module `For studies purposes only`
An implementation from micrograd engine.
Credit to Karpathy for the original implementation.
"""

import math
import random

from graphviz import Digraph


class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []


class Value:
    """A class that represents a value in the computation graph.

    Attributes:
        data (float): The numerical value.
        grad (float): The gradient of the value.
        _backward (function): A function to compute the gradient.
        _prev (set): A set of previous Value instances in the computation graph.
        _op (str): The operation that produced this value.
    """

    def __init__(self, data, _children: tuple = (), _op: str = "", label: str = ""):
        self.data = data  # Given numerical value
        self.grad = 0.0  # Gradient initialized to zero
        self._backward = lambda: None  # Placeholder for backward function
        self._prev = set(_children)  # Set of previous Value instances
        self._op = _op  # Operation that produced this value
        self.label = label

    def __add__(self, other):
        # This is the implementation of
        # a + b -> a.__add__(b)
        # This is used when adding bias
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        # This is used when w * x + b
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def backward(self):

        # Topological order
        topo: list[Value] = []
        visited = set()

        def build_topo(v: Value) -> None:
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        # build topo
        build_topo(self)

        # Iterate through all nodes
        self.grad = 1
        for v in reversed(topo):
            v._backward()

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1 - t**2) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), "ReLU")

        # ReLU(x) = max(0, x)
        # d/dx(ReLU(x)) = 0  if x < 0
        #                 1  if x > 0
        #                 undefined at x = 0 (we use 0, like PyTorch)
        def _backward():
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward

        return out

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self,), "exp")

        # d/dx(e^x) = e^x
        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    # Extra functions
    def __rmul__(self, other):  # reverse multiplication
        # 2 * a -> int can't multiply a Value, so Python falls back to a.__rmul__(2)
        return self * other

    def __radd__(self, other):  # reverse addition
        # 1 + a -> int can't add a Value, so Python falls back to a.__radd__(1)
        return self + other

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f"**{other}")

        # The derivative of the power x^n is given by
        # d/(d x)(x^n) = n x^(n - 1).
        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad

        out._backward = _backward
        return out

    def __truediv__(self, other):  # self / other
        return self * other**-1  # __pow__ then __mul__

    def __rtruediv__(self, other):  # other / self
        return other * self**-1

    def __neg__(self):  # -self
        return self * -1

    def __sub__(self, other):  # self - other
        return self + (-other)

    def __rsub__(self, other):  # other - self
        return other + (-self)

    def __repr__(self) -> str:
        return f"Value(label={self.label}, data={self.data}, grad={self.grad})"


class NnGraph:
    """A Class to show the NN Graph bias and Backpropagation"""

    def __init__(self):
        pass

    @staticmethod
    def traces(root: Value):
        nodes, edges = set(), set()

        def _build(v):
            if v not in nodes:
                nodes.add(v)
                for child in v._prev:
                    edges.add((child, v))
                    _build(child)

        _build(root)
        return nodes, edges

    @classmethod
    def draw_dot(cls, root: Value):
        dot = Digraph(format="svg", graph_attr={"rankdir": "LR"})

        nodes, edges = cls.traces(root)
        for n in nodes:
            uid = str(id(n))
            dot.node(
                name=uid,
                label="{ %s | data %.4f | grad %.4f }" % (n.label, n.data, n.grad),
                shape="record",
            )
            if n._op:
                # if value is an operation result, create a node for it
                dot.node(name=uid + n._op, label=n._op)
                dot.edge(uid + n._op, uid)  # Connect this node

        for n1, n2 in edges:
            dot.edge(str(id(n1)), str(id(n2)) + n2._op)
        return dot


class Neuron(Module):
    """Base class to create Neuron"""

    def __init__(self, nin: int):
        # Create random weights and a random bias
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        # w * x + b
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        #     └──── Σ wi*xi ────────────────────────┘   └ + b ┘
        out = act.tanh()
        return out

    def parameters(self):
        # Joins the weights with the bias into one list, e.g. Neuron(3):
        #
        #   self.w   = [ w0, w1, w2 ]    one weight per input  -> nin = 3
        #   [self.b] = [ b ]             one bias per neuron   -> + 1
        #                \_____|_____/
        #                      |
        #   [w0, w1, w2, b]  -> 3 + 1 = 4 parameters (nin + 1)
        return self.w + [self.b]

    def __str__(self) -> str:
        lines = [f"Neuron(nin={len(self.w)})"]
        lines += [f"  w{i} = {w.data:+.4f}" for i, w in enumerate(self.w)]
        lines.append(f"  b  = {self.b.data:+.4f}")
        return "\n".join(lines)


class Layer(Module):
    """Base Class Layer"""

    def __init__(self, nin: int, nout: int):
        # nin:  input dimension (features per sample). For data shaped [n, m]
        #       (n samples, m features), nin = m.
        # nout: number of neurons in the layer, which is also the output dimension.
        self._nin = nin
        self._nout = nout
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        # Feed the input x to every neuron in the layer
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        # Flattens every neuron's parameters into one list, e.g. Layer(2, 3):
        #
        #   self.neurons = [ N0,           N1,           N2          ]
        #                    |             |             |
        #   parameters()   [w0, w1, b]   [w0, w1, b]   [w0, w1, b]
        #                    \_____________|_____________/
        #                                  |
        #   [N0.w0, N0.w1, N0.b,  N1.w0, N1.w1, N1.b,  N2.w0, N2.w1, N2.b]  -> 9 parameters
        return [p for neuron in self.neurons for p in neuron.parameters()]

    def __repr__(self) -> str:
        return f"Layer(nin={self._nin}, nout={self._nout})"

    def __str__(self) -> str:
        lines = [repr(self)]
        for i, n in enumerate(self.neurons):
            neuron_lines = str(n).split("\n")
            lines.append(f"  [{i}] " + neuron_lines[0])
            lines += ["      " + line for line in neuron_lines[1:]]
        return "\n".join(lines)


class MLP(Module):
    """Base class for Multi Layer Perceptron"""

    def __init__(self, nin: int, nouts: list[int]):
        # sz joins the input size with every layer size, e.g. MLP(3, [4, 5, 2, 1]):
        #
        #   sz = [ 3,  4,  5,  2,  1 ]
        #          \__/
        #   i=0    Layer(3, 4)
        #              \__/
        #   i=1        Layer(4, 5)
        #                  \__/
        #   i=2            Layer(5, 2)
        #                      \__/
        #   i=3                Layer(2, 1)
        #
        # Each layer's nout is the next layer's nin.
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i + 1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            # the output of the previous layer is the input of the next one
            x = layer(x)
        return x

    def parameters(self):
        #   self.layers   = [ Layer(2, 3),                           Layer(3, 1)      ]
        #                     |                                      |
        #   self.neurons  = [ N0,          N1,          N2      ]    [ N0             ]
        #                     |            |            |              |
        #   parameters()    [w0, w1, b]  [w0, w1, b]  [w0, w1, b]    [w0, w1, w2, b]
        #                     \____________|____________/              |
        #                                  |                           |
        #   per layer         3 * (2 + 1) = 9                        1 * (3 + 1) = 4
        #                                  \___________________________/
        #                                                |
        #   [9 from L0, 4 from L1]  -> 13 parameters (a sum, not a product)

        return [p for layer in self.layers for p in layer.parameters()]

    def __str__(self) -> str:
        lines = [repr(self)]
        for i, layer in enumerate(self.layers):
            layer_lines = str(layer).split("\n")
            lines.append(f"  ({i}) " + layer_lines[0])
            lines += ["      " + line for line in layer_lines[1:]]
        lines.append(f"  total parameters: {len(self.parameters())}")
        return "\n".join(lines)
