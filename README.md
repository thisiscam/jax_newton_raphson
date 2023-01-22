# jax_newton_raphson
A simple Newton-Raphson optimizer in JAX.

## Install

```bash
pip install git+https://github.com/thisiscam/jax_newton_raphson
```

## Usage

```python
import collections
import jax_newton_raphson as jnr

Params = collections.namedtuple("Params", "x y")


def f(params: Params):
  return (params.x**2 + params.y**2)


print(jnr.minimize(f, initial_guess=Params(-0.1, 0.1)))
```