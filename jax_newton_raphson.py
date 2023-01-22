"""A simple Newton-Raphson optimizer in JAX."""

from typing import Callable, NamedTuple, Tuple

import collections
import functools

import jax
import jax.lax
import jax.numpy as jnp
import jax.scipy
import jax.flatten_util

import chex


def _value_and_jacfwd(f, x, has_aux=False):
  pushfwd = functools.partial(jax.jvp, f, (x,), has_aux=has_aux)
  basis = (jnp.eye(x.size, dtype=x.dtype),)
  out_axes = (None, -1, None) if has_aux else (None, -1)
  return jax.vmap(pushfwd, out_axes=out_axes)(basis)


def _jacrev_and_value(f, x):
  y, pullback = jax.vjp(f, x)
  basis = jnp.eye(y.size, dtype=y.dtype)
  jac = jax.vmap(pullback)(basis.ravel())
  return jac, y


def _value_jac_and_hessian(f):
  """Compute value, jacobian and hessian of a function in one go.

  For simplicity, we assume that `f` has vector input and scalar output.

  Args:
    f: the function to compute the value, jacobian and hessian of.

  Returns:
    A function that takes a vector and returns a tuple of (value, jacobian,
    hessian).
  """

  def wrapped(x):
    (jac,), (hessian,), value = _value_and_jacfwd(
        functools.partial(_jacrev_and_value, f),
        x,
        has_aux=True,
    )
    return value, jac.reshape(jac.size), hessian.reshape(jac.size, jac.size)

  return wrapped


def _cholesky_with_added_identity(mat: chex.Array,
                                  cho_beta: float,
                                  cho_tau_factor: float = 2.) -> chex.Array:
  """Compute the Cholesky factorization of a matrix with added scaled identity.

  See "Numerical Optimization" by Nocedal and Wright, Algorithm 3.3.

  Args:
    mat: the matrix to compute the Cholesky factorization of.
    cho_beta: the heuristically chosen minimal factor to add to the diagonal.
    cho_tau_factor: the factor by which to increase the diagonal if the matrix
      is not positive definite.

  Returns:
    The Cholesky factorization of `mat + identity`.
  """
  LoopState = collections.namedtuple("LoopState", "tau cho_factor")

  min_diag = jnp.min(jnp.diagonal(mat))
  tau0 = jnp.where(min_diag > 0, jnp.zeros_like(min_diag), cho_beta - min_diag)

  def loop_body(state: LoopState) -> LoopState:
    cho_factor = jax.scipy.linalg.cholesky(mat +
                                           jnp.eye(mat.shape[0]) * state.tau)
    tau = jnp.maximum(cho_tau_factor * state.tau, cho_beta)
    return LoopState(tau, cho_factor)

  loop_state = jax.lax.while_loop(
      lambda state: jnp.logical_not(jnp.all(jnp.isfinite(state.cho_factor))),
      loop_body,
      LoopState(tau0, jnp.full_like(mat, jnp.nan)),
  )
  return loop_state.cho_factor


class NewtonRaphsonResult(NamedTuple):
  """The result of the Newton-Raphson minimizer."""
  guess: chex.ArrayTree
  fnval: chex.Scalar
  jac: chex.Array
  hessian: chex.Array
  step: int
  converged: bool


def minimize(
    fn: Callable[[chex.ArrayTree], chex.Scalar],
    initial_guess: chex.ArrayTree,
    atol=1e-05,
    rtol=1e-08,
    line_search_factor: float = 0.5,
    maxiters: int = 10,
    cho_beta: float = 1e-08,
    cho_tau_factor: float = 2.,
) -> NewtonRaphsonResult:
  """Newton-Raphson minization for strictly convex function in JAX, jit-able.

  Currently assumes that the hessian for all steps are positive definite.

  Args:
    fn: the function to minimize. The function must take a vector as input and
      return a scalar.
    initial_guess: the initial guess.
    atol: the absolute tolerance for convergence test.
    rtol: the relative tolerance for convergence test.
    line_search_factor: the factor by which to reduce the step size if the
      function value does not decrease.
    maxiters: maximum number of optimizer iterations; note that this includes
      the number of line search steps.
    cho_beta, cho_tau_factor: for modified cholesky to ensure positive 
    definiteness of the hessian. See `_cholesky_with_added_identity`.

  Returns:
    The optimizer result.
  """
  chex.assert_shape(jax.eval_shape(fn, initial_guess), tuple())

  initial_guess_flat, guess_unraveler = jax.flatten_util.ravel_pytree(
      initial_guess)
  x_dim = initial_guess_flat.size

  def flatten_fn(guess):
    return fn(guess_unraveler(guess))

  value_jac_and_hessian_fn = _value_jac_and_hessian(flatten_fn)
  LoopState = collections.namedtuple(
      "LoopState", "guess new_guess fnval jac hessian step converged")
  WorkState = collections.namedtuple(
      "WorkState",
      "new_fnval new_jac new_hessian cho_factor is_finite fnval_decreased")

  def _line_search(args):
    """Decrease step size by line_search_factor."""
    loop_state, _ = args
    new_guess_ = (loop_state.new_guess -
                  loop_state.guess) * line_search_factor + loop_state.guess
    return loop_state._replace(new_guess=new_guess_)

  def _newton_guess_update(args):
    loop_state, working_state = args
    u = jax.scipy.linalg.cho_solve((working_state.cho_factor, False),
                                   working_state.new_jac)
    new_guess_ = loop_state.new_guess - u
    return loop_state._replace(fnval=working_state.new_fnval,
                               guess=loop_state.new_guess,
                               new_guess=new_guess_)

  def _do_work(args: Tuple[LoopState, WorkState]) -> LoopState:
    """Perform an update step if necessary."""
    _, work_state = args
    dont_need_line_search = work_state.is_finite
    dont_need_line_search = jnp.logical_and(dont_need_line_search,
                                            work_state.fnval_decreased)
    loop_state = jax.lax.cond(dont_need_line_search,
                              _newton_guess_update,
                              _line_search,
                              operand=args)
    return loop_state._replace(step=loop_state.step + 1)

  def _do_converged(args: Tuple[LoopState, WorkState]) -> LoopState:
    loop_state, working_state = args
    return loop_state._replace(fnval=working_state.new_fnval,
                               guess=loop_state.new_guess,
                               jac=working_state.new_jac,
                               hessian=working_state.new_hessian,
                               converged=True)

  def loop_body(loop_state: LoopState) -> LoopState:
    """Newton-Raphson loop body."""
    new_fnval, new_jac, new_hessian = value_jac_and_hessian_fn(
        loop_state.new_guess)

    # If hesseian is not even finite, we just revert gradient descent
    # We do this before the cholesky factorization to avoid infinite looping
    # in the factorization
    new_hessian = jnp.where(jnp.all(jnp.isfinite(new_hessian)), new_hessian,
                            jnp.eye(new_hessian.shape[0]))
    cho_factor = _cholesky_with_added_identity(new_hessian, cho_beta,
                                               cho_tau_factor)
    is_finite = jnp.logical_and(jnp.all(jnp.isfinite(new_fnval)),
                                jnp.all(jnp.isfinite(new_fnval)))
    converged = jnp.logical_and(
        is_finite,
        jnp.allclose(new_fnval, loop_state.fnval, atol=atol, rtol=rtol))
    loop_state = loop_state._replace(converged=converged)

    fnval_decreased = loop_state.fnval > new_fnval
    working_state = WorkState(new_fnval, new_jac, new_hessian, cho_factor,
                              is_finite, fnval_decreased)

    return jax.lax.cond(jnp.logical_and(converged, loop_state.step > 0),
                        _do_converged,
                        _do_work,
                        operand=(loop_state, working_state))

  def loop_cond(loop_state: LoopState) -> bool:
    return jnp.logical_or(
        loop_state.step == 0,  # Run at least one iteration
        jnp.logical_and(loop_state.step < maxiters,
                        jnp.logical_not(loop_state.converged)))

  initial_state = LoopState(initial_guess_flat, initial_guess_flat, jnp.inf,
                            jnp.full_like(initial_guess_flat, jnp.inf),
                            jnp.empty((x_dim, x_dim)), 0, False)

  loop_state = jax.lax.while_loop(loop_cond, loop_body, initial_state)

  def do_recover_last(loop_state: LoopState) -> LoopState:
    fnval, jac, hessian = value_jac_and_hessian_fn(loop_state.guess)
    return loop_state._replace(fnval=fnval, jac=jac, hessian=hessian)

  loop_state = jax.lax.cond(loop_state.converged,
                            lambda state: state,
                            do_recover_last,
                            operand=loop_state)

  return NewtonRaphsonResult(guess_unraveler(loop_state.guess),
                             loop_state.fnval, loop_state.jac,
                             loop_state.hessian, loop_state.step,
                             loop_state.converged)
