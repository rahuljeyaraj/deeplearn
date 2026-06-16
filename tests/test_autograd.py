import pytest
import numpy as np

class TestValue:
    def test_add_fwd(self):
        from deeplearn.autograd import Value
        a = Value(2.0)
        b = Value(3.0)
        c = a + b
        assert c.data == 5.0 