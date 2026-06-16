import math


class Value:
    def __init__(self, data: float, _children: tuple = (), _op: str = ""):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None  # filled in by each op
        self._prev = set(_children)
        self._op = _op  # for debugging: shows what created this node

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data

        out._backward = _backward
        return out

    def __pow__(self, exp):
        out = Value(self.data**exp, (self,), f"**{exp}")

        def _backward():
            self.grad += out.grad * exp * (self.data ** (exp - 1))

        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __radd__(self, other):
        return self.__add__(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __rsub__(self, other):
        return other + (-self)

    def backward(self):
        topo, visited = [], set()

        def build_topo(value):
            if value in visited:
                return
            visited.add(value)
            for p in value._prev:
                build_topo(p)
            topo.append(value)

        build_topo(self)
        self.grad = 1.0
        for value in reversed(topo):
            value._backward()

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"
