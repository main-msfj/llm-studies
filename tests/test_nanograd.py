import torch

from src.nanograd import Value


def test_add_same_node_twice():
    a = Value(3.0)
    b = a + a
    b.backward()

    assert b.data == 6.0
    assert a.grad == 2.0


def test_mul_same_node_twice():
    a = Value(3.0)
    b = a * a
    b.backward()

    assert b.data == 9.0
    assert a.grad == 6.0


def test_neuron_grads_match_pytorch():

    # Values
    vals = [1.0, 2.0, 3.0, 4.0, 6.881373587019532]

    # NanoGrad
    x1, x2 = Value(vals[0]), Value(vals[1])
    w1, w2 = Value(vals[2]), Value(vals[3])
    b = Value(vals[-1])
    o = (x1 * w1 + x2 * w2 + b).tanh()
    o.backward()

    # Pytorch
    tx1, tx2, tw1, tw2, tb = (torch.tensor(v, dtype=torch.double, requires_grad=True) for v in vals)
    to = torch.tanh(tx1 * tx2 + tw1 * tw2 + tb)
    to.backward()

    assert abs(o.data - to.item()) < 1e-6
    for mine, ref in [(x1, tx1), (x2, tx2), (w1, tw1), (w2, tw2), (b, tb)]:
        assert abs(mine.grad - ref.grad.item()) < 1e-6


def test_reused_node_matches_pytorch():

    # values
    _a, _b = -2.0, 3.0

    # Nanograd
    a, b = Value(_a), Value(_b)
    d = a * b + a
    e = d * a
    e.backward()

    # Pytorch
    ta = torch.tensor(_a, dtype=torch.double, requires_grad=True)
    tb = torch.tensor(_b, dtype=torch.double, requires_grad=True)
    td = ta * tb + ta
    te = td * ta
    te.backward()

    assert abs(e.data - te.item()) < 1e-6
    assert abs(a.grad - ta.grad.item()) < 1e-6
    assert abs(b.grad - tb.grad.item()) < 1e-6


def test_more_ops():

    a = Value(-4.0)
    b = Value(2.0)
    c = a + b
    d = a * b + b**3
    c += c + 1
    c += 1 + c + (-a)
    d += d * 2 + (b + a).relu()
    d += 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g += 10.0 / f
    g.backward()
    amg, bmg, gmg = a, b, g

    a = torch.Tensor([-4.0]).double()
    b = torch.Tensor([2.0]).double()
    a.requires_grad = True
    b.requires_grad = True
    c = a + b
    d = a * b + b**3
    c = c + c + 1
    c = c + 1 + c + (-a)
    d = d + d * 2 + (b + a).relu()
    d = d + 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g = g + 10.0 / f
    g.backward()
    apt, bpt, gpt = a, b, g

    tol = 1e-6

    assert abs(gmg.data - gpt.data.item()) < tol
    assert abs(amg.grad - apt.grad.item()) < tol
    assert abs(bmg.grad - bpt.grad.item()) < tol
