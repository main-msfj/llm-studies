"""Nano Grad Engine module `For studies purposes only`
An implementation from micrograd engine.
Credit to Karpathy for the original implementation.
"""

import math

from graphviz import Digraph


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
        # This is the implentation of
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
        # This is used when W-x + B
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        x = self.data
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1 - t**2) * out.grad

        out._backward = _backward
        return out


class NnGraph:
    """A Class to show the NN Graph bias and BackProgation"""

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
