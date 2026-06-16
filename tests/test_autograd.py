import numpy as np
import pytest

from deeplearn.autograd import Value  # Adjust package namespace if necessary


class TestValue:
    def test_addition(self):
        """Validates forward and localized backward operations for addition."""
        a = Value(2.0)
        b = Value(-3.0)

        # Forward pass
        c = a + b
        assert np.isclose(c.data, -1.0)

        # Local backward pass
        c.grad = 1.0
        c._backward()
        assert np.isclose(a.grad, 1.0)
        assert np.isclose(b.grad, 1.0)

    def test_multiplication(self):
        """Validates forward and localized backward operations for multiplication."""
        a = Value(3.0)
        b = Value(4.0)

        # Forward pass
        c = a * b
        assert np.isclose(c.data, 12.0)

        # Local backward pass
        c.grad = 1.0
        c._backward()
        assert np.isclose(a.grad, 4.0)  # dc/da = b
        assert np.isclose(b.grad, 3.0)  # dc/db = a

    def test_power(self):
        """Validates forward and localized backward operations for powers."""
        a = Value(3.0)

        # Forward pass (3^3 = 27)
        c = a**3
        assert np.isclose(c.data, 27.0)

        # Local backward pass
        c.grad = 1.0
        c._backward()
        assert np.isclose(a.grad, 27.0)  # dc/da = 3 * a^2 = 3 * 9 = 27

    def test_subtraction_and_negation(self):
        """Validates that negation and subtraction compose correctly."""
        a = Value(5.0)
        b = Value(2.0)

        # Forward passes
        assert np.isclose((-a).data, -5.0)
        c = a - b
        assert np.isclose(c.data, 3.0)

        # Local backward verification
        c.grad = 1.0
        c._backward()
        # Subtraction acts as a + (-b), so gradients downstream should inherit 1.0
        # and be propagated through the internal __add__ operation
        assert np.isclose(a.grad, 1.0)

    def test_reverse_operations(self):
        """Validates right-side reflection methods (__radd__, __rmul__, __rsub__) when operating with scalars."""
        a = Value(2.0)

        # __radd__: 5 + a
        b = 5 + a
        assert np.isclose(b.data, 7.0)

        # __rmul__: 3 * a
        c = 3 * a
        assert np.isclose(c.data, 6.0)

        # __rsub__: 10 - a
        d = 10 - a
        assert np.isclose(d.data, 8.0)

    def test_full_graph_backpropagation(self):
        """Tests the full topological sort and global `.backward()` graph processing.

        This sets up a deeper expression graph containing addition, subtraction,
        multiplication, and power operations to verify execution order and gradient accumulation.
        Expression: f = (a * b + c) ** 2 - d
        """
        a = Value(2.0)
        b = Value(-3.0)
        c = Value(10.0)
        d = Value(4.0)

        # Forward Pass
        # e = (2.0 * -3.0) + 10.0 = -6.0 + 10.0 = 4.0
        e = a * b + c
        # f = 4.0**2 - 4.0 = 16.0 - 4.0 = 12.0
        f = e**2 - d

        assert np.isclose(f.data, 12.0)

        # Execute full autograd backpropagation
        f.backward()

        # Analytical Gradients Verification:
        # df/dd = -1.0
        assert np.isclose(d.grad, -1.0)

        # df/de = 2 * e = 2 * 4.0 = 8.0
        # df/dc = df/de * de/dc = 8.0 * 1.0 = 8.0
        assert np.isclose(c.grad, 8.0)

        # df/db = df/de * de/db = 8.0 * a = 8.0 * 2.0 = 16.0
        assert np.isclose(b.grad, 16.0)

        # df/da = df/de * de/da = 8.0 * b = 8.0 * -3.0 = -24.0
        assert np.isclose(a.grad, -24.0)

    def test_gradient_accumulation(self):
        """Validates that reusing the same Value node accumulated gradients properly rather than overwriting.

        Expression: b = a + a + a
        """
        a = Value(3.0)
        b = a + a + a

        assert np.isclose(b.data, 9.0)

        b.backward()
        # db/da should be 1 + 1 + 1 = 3
        assert np.isclose(a.grad, 3.0)
