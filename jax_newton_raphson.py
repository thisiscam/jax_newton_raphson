"""A Newton-Raphson optimizer in JAX."""

from typing import Callable, NamedTuple, Optional

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


def _value_jac_and_hessian(f, x):
  """Compute value, jacobian and hessian of a function in one go.

  For simplicity, we assume that `f` has vector input and scalar output.

  Args:
    f: the function to compute the value, jacobian and hessian of.

  Returns:
    A function that takes a vector and returns a tuple of (value, jacobian,
    hessian).
  """

  (jac,), (hessian,), value = _value_and_jacfwd(
      functools.partial(_jacrev_and_value, f),
      x,
      has_aux=True,
  )
  return value, jac.reshape(jac.size), hessian.reshape(jac.size, jac.size)


class CholeskyWithAddedIdentityLoopState(NamedTuple):
  tau: float
  cho_factor: chex.Array


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

  min_diag = jnp.min(jnp.diagonal(mat))
  tau0 = jnp.where(min_diag > 0, jnp.zeros_like(min_diag), cho_beta - min_diag)

  def cond_fn(state: CholeskyWithAddedIdentityLoopState) -> bool:
    return ~jnp.all(jnp.isfinite(state.cho_factor))

  def loop_body(
      state: CholeskyWithAddedIdentityLoopState
  ) -> CholeskyWithAddedIdentityLoopState:
    cho_factor = jax.scipy.linalg.cholesky(
        mat + jnp.eye(mat.shape[0], dtype=mat.dtype) * state.tau)
    tau = jnp.maximum(cho_tau_factor * state.tau, cho_beta)
    return CholeskyWithAddedIdentityLoopState(tau, cho_factor)

  loop_state = jax.lax.while_loop(
      cond_fn,
      loop_body,
      CholeskyWithAddedIdentityLoopState(tau0, jnp.full_like(mat, jnp.nan)),
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
    msgs = {
        0: "converged",
        1: "maximum number of iterations reached",
        2: "maximum number of line search steps reached",
    }
    return msgs.get(self.status, "unknown status")

  @property
  def converged(self):
    return self.status == 0


class LoopState(NamedTuple):
  """The state of the Newton-Raphson minimizer loop."""
  guess: chex.Array
  new_guess: chex.Array
  fnval: float
  jac: chex.Array
  hessian: chex.Array
  newton_step: int
  ls_step: int
  converged: bool


class WorkingState(NamedTuple):
  """For internal temporary storage inside the loop body."""
  new_fnval: float
  new_jac: chex.Array
  new_hessian: chex.Array
  is_finite: bool
  fnval_decreased: bool


class Parameters(NamedTuple):
  line_search_factor: float
  cho_beta: float
  cho_tau_factor: float


def _ls_step(args):
  """Backtracking linesearch: Decrease step size by line_search_factor."""
  params, loop_state, _ = args
  new_guess_ = (loop_state.new_guess -
                loop_state.guess) * params.line_search_factor + loop_state.guess
  return loop_state._replace(new_guess=new_guess_,
                             ls_step=loop_state.ls_step + 1)


def _newton_guess_update(args):
  """Update the guess using the Newton-Raphson step."""
  params, loop_state, working_state = args
  # If hesseian is not even finite, we just revert to gradient descent
  # We do this before the cholesky factorization to avoid infinite looping
  # in the factorization
  new_hessian = working_state.new_hessian
  new_hessian = jnp.where(
      jnp.all(jnp.isfinite(new_hessian)), new_hessian,
      jnp.eye(new_hessian.shape[0], dtype=new_hessian.dtype))
  cho_factor = _cholesky_with_added_identity(new_hessian, params.cho_beta,
                                             params.cho_tau_factor)
  u = jax.scipy.linalg.cho_solve((cho_factor, False), working_state.new_jac)
  new_guess_ = loop_state.new_guess - u
  return loop_state._replace(fnval=working_state.new_fnval,
                             guess=loop_state.new_guess,
                             new_guess=new_guess_,
                             newton_step=loop_state.newton_step + 1,
                             ls_step=jnp.array(0, dtype=jnp.int32))


def _do_line_search_or_newton_step(args) -> LoopState:
  """Perform an update step if necessary."""
  _, _, work_state = args
  dont_need_line_search = work_state.is_finite & work_state.fnval_decreased
  return jax.lax.cond(dont_need_line_search,
                      _newton_guess_update,
                      _ls_step,
                      operand=args)


def _do_converged(args) -> LoopState:
  """Return the converged state."""
  _, loop_state, working_state = args
  return loop_state._replace(fnval=working_state.new_fnval,
                             guess=loop_state.new_guess,
                             jac=working_state.new_jac,
                             hessian=working_state.new_hessian,
                             converged=True)


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
    parameter_mask: Optional[chex.ArrayTree] = None,
) -> NewtonRaphsonResult:
  """Newton-Raphson minization in JAX, batchable and jit-able.

  This function implements a variant of Newton-Raphson minimization that
  includes the following modifications to the vanilla algorithm:
  1. Cholesky factorization on modified hessian to ensure positive definiteness.
  2. Backtracking line search to ensure that the function value decreases
  sufficiently at each step, that is, each newton step satisfies the Armijo
  condition. With these modifications, the algorithm is guaranteed to converge
  to a global minimum on a convex function with sufficiently large values of 
  `maxiter` and `maxls`.

  Note that this implementation sychronizes the calculations of the
  function's value, jacobian and hessian, when user invokes it in
  batch mode via `jax.vmap`. In particular, this means that for a single
  iteration of the optimizer loop, the algorithm can either take a line search
  step or a Newton step. In principle, this minimizes thread divergence in
  batch mode. However, the tradeoff is that the algorithm does one redundant
  hessian evaluation for each backtracking line search step.

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
    parameter_mask: a pytree of boolean arrays with the same structure as the
      input to `fn`. If provided, the optimized function `fn` must depend only
      on the parameters where the corresponding mask is True. This is useful
      for batching the optimization of multiple functions with different sets
      of parameters. In that case, user may pad the parameters of each function
      with zeros and set the corresponding mask to False.

  Returns:
    The optimizer result.
  """
  chex.assert_shape(jax.eval_shape(fn, initial_guess), tuple())

  params = Parameters(line_search_factor, cho_beta, cho_tau_factor)

  initial_guess_flat, guess_unraveler = jax.flatten_util.ravel_pytree(
      initial_guess)
  x_dim = initial_guess_flat.size

  if parameter_mask is not None:
    chex.assert_trees_all_equal_shapes(initial_guess, parameter_mask)
    parameter_mask, _ = jax.flatten_util.ravel_pytree(parameter_mask)

  def flatten_fn(guess):
    return fn(guess_unraveler(guess))

  def loop_body(loop_state: LoopState) -> LoopState:
    """Newton-Raphson loop body."""
    new_fnval, new_jac, new_hessian = _value_jac_and_hessian(
        flatten_fn, loop_state.new_guess)

    if parameter_mask is not None:
      avg_diag = (jnp.sum(jnp.abs(jnp.diagonal(new_hessian))) /
                  jnp.sum(parameter_mask))
      diag_idx = jnp.diag_indices(new_hessian.shape[0])
      new_hessian = new_hessian.at[diag_idx].add(avg_diag *
                                                 (1 - parameter_mask))

    is_finite = (jnp.all(jnp.isfinite(new_fnval)) &
                 jnp.all(jnp.isfinite(new_jac)))
    converged = (is_finite & jnp.allclose(
        new_fnval, loop_state.fnval, atol=atol, rtol=rtol))

    # Check if new guess satisfies the Armijo condition
    fnval_decreased = (
        new_fnval <= loop_state.fnval +
        loop_state.jac @ (loop_state.new_guess - loop_state.guess))

    working_state = WorkingState(new_fnval, new_jac, new_hessian, is_finite,
                                 fnval_decreased)

    return jax.lax.cond(
        (converged & ((loop_state.newton_step > 0) | (loop_state.ls_step > 0))),
        _do_converged,
        _do_line_search_or_newton_step,
        operand=(params, loop_state, working_state),
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
      jnp.empty((x_dim, x_dim), dtype=initial_guess_flat.dtype),
      jnp.array(0, dtype=jnp.int32),
      jnp.array(0, dtype=jnp.int32),
      jnp.array(False, dtype=bool),
  )

  loop_state = jax.lax.while_loop(loop_cond, loop_body, initial_state)

  fnval, jac, hessian = _value_jac_and_hessian(flatten_fn, loop_state.guess)
  loop_state = loop_state._replace(fnval=fnval, jac=jac, hessian=hessian)

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
