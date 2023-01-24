"""A Newton-Raphson optimizer in JAX."""

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
      lambda state: ~jnp.all(jnp.isfinite(state.cho_factor)),
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
  status: int

  @property
  def msg(self):
    if self.status == 0:
      return "Converged"
    elif self.status == 1:
      return "Maximum number of iterations reached"
    elif self.status == 2:
      return "Maximum number of line search steps reached"
    else:
      return "Unknown status"


def minimize(
    fn: Callable[[chex.ArrayTree], chex.Scalar],
    initial_guess: chex.ArrayTree,
    atol=1e-05,
    rtol=1e-08,
    line_search_factor: float = 0.5,
    maxiter: int = 10,
    maxls: int = 10,
    cho_beta: float = 1e-08,
    cho_tau_factor: float = 2.,
) -> NewtonRaphsonResult:
  """Newton-Raphson minization in JAX, jit-able.

  This function implements a variant of Newton-Raphson minimization that
  includes the following modifications to the vanilla algorithm:
  1. Cholesky factorization on modified hessian to ensure positive definiteness.
  2. Backtracking line search to ensure that the function value decreases
  sufficiently at each step, that is, the step satisfies the Armijo condition.
  With these modifications, the algorithm is guaranteed to converge to a global
  minimum on a convex function.

  Note that this implementation to sychronizes the calculations of the
  function's value, jacobian and hessian, when user invokes it in
  batch mode via `jax.vmap`. In particular, this means that for a single
  iteration of the optimizer loop, the algorithm can either take a line search
  step or a Newton step.

  Args:
    fn: the function to minimize. The function must take a pytree as input and
      return a scalar.
    initial_guess: the initial guess.
    atol: the absolute tolerance for convergence test.
    rtol: the relative tolerance for convergence test.
    line_search_factor: the factor by which to reduce the step size if the
      function value does not decrease.
    maxiter: maximum number of optimizer iterations; note that this includes
      the number of line search steps.
    maxls: maximum number of line search steps.
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

  # State for newton-raphson loop
  LoopState = collections.namedtuple(
      "LoopState",
      "guess new_guess fnval jac hessian newton_step ls_step converged")

  # For internal temporay storage inside the loop body
  WorkingState = collections.namedtuple(
      "WorkingState",
      "new_fnval new_jac new_hessian cho_factor is_finite fnval_decreased")

  def _ls_step(args):
    """Backtracking linesearch: Decrease step size by line_search_factor."""
    loop_state, _ = args
    new_guess_ = (loop_state.new_guess -
                  loop_state.guess) * line_search_factor + loop_state.guess
    return loop_state._replace(new_guess=new_guess_,
                               ls_step=loop_state.ls_step + 1)

  def _newton_guess_update(args):
    """Update the guess using the Newton-Raphson step."""
    loop_state, working_state = args
    u = jax.scipy.linalg.cho_solve((working_state.cho_factor, False),
                                   working_state.new_jac)
    new_guess_ = loop_state.new_guess - u
    return loop_state._replace(fnval=working_state.new_fnval,
                               guess=loop_state.new_guess,
                               new_guess=new_guess_,
                               newton_step=loop_state.newton_step + 1,
                               ls_step=0)

  def _do_line_search_or_newton_step(
      args: Tuple[LoopState, WorkingState]) -> LoopState:
    """Perform an update step if necessary."""
    _, work_state = args
    dont_need_line_search = work_state.is_finite
    dont_need_line_search = (dont_need_line_search & work_state.fnval_decreased)
    return jax.lax.cond(dont_need_line_search,
                        _newton_guess_update,
                        _ls_step,
                        operand=args)

  def _do_converged(args: Tuple[LoopState, WorkingState]) -> LoopState:
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

    # If hesseian is not even finite, we just revert to gradient descent
    # We do this before the cholesky factorization to avoid infinite looping
    # in the factorization
    new_hessian = jnp.where(jnp.all(jnp.isfinite(new_hessian)), new_hessian,
                            jnp.eye(new_hessian.shape[0]))
    cho_factor = _cholesky_with_added_identity(new_hessian, cho_beta,
                                               cho_tau_factor)
    is_finite = (jnp.all(jnp.isfinite(new_fnval)) &
                 jnp.all(jnp.isfinite(new_jac)))
    converged = (is_finite & jnp.allclose(
        new_fnval, loop_state.fnval, atol=atol, rtol=rtol))
    # Check if new guess satisfies the Armijo condition
    fnval_decreased = (
        new_fnval <= loop_state.fnval +
        loop_state.jac @ (loop_state.new_guess - loop_state.guess))

    working_state = WorkingState(new_fnval, new_jac, new_hessian, cho_factor,
                                 is_finite, fnval_decreased)

    return jax.lax.cond(
        (converged & ((loop_state.newton_step > 0) | (loop_state.ls_step > 0))),
        _do_converged,
        _do_line_search_or_newton_step,
        operand=(loop_state, working_state),
    )

  def loop_cond(loop_state: LoopState):
    """Run at least one iteration, stop if converged or maxiter reached."""
    return (((loop_state.newton_step == 0) & (loop_state.ls_step == 0)) |
            ((loop_state.newton_step < maxiter + 1) &
             (loop_state.ls_step < maxls + 1) & ~loop_state.converged))

  initial_state = LoopState(
      initial_guess_flat,
      initial_guess_flat,
      jnp.inf,
      jnp.zeros_like(initial_guess_flat),
      jnp.empty((x_dim, x_dim)),
      jnp.array(0),
      jnp.array(0),
      jnp.array(False),
  )

  loop_state = jax.lax.while_loop(loop_cond, loop_body, initial_state)

  def do_recover_last(loop_state: LoopState) -> LoopState:
    fnval, jac, hessian = value_jac_and_hessian_fn(loop_state.guess)
    return loop_state._replace(fnval=fnval, jac=jac, hessian=hessian)

  loop_state = jax.lax.cond(loop_state.converged,
                            lambda state: state,
                            do_recover_last,
                            operand=loop_state)

  return NewtonRaphsonResult(
      guess_unraveler(loop_state.guess),
      loop_state.fnval,
      loop_state.jac,
      loop_state.hessian,
      loop_state.newton_step,
      jnp.where(
          loop_state.newton_step >= maxiter, 1,
          jnp.where(loop_state.ls_step >= maxls, 2,
                    jnp.where(loop_state.converged, 0, 3))),
  )
