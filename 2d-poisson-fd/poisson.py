import jax
import jax.numpy as jnp
import lineax as lx
import equinox as eqx

from jaxtyping import Array, Float
from collections.abc import Callable

N = 100
h = 1/N
x = jnp.linspace(0, 1, N+1)
y = jnp.linspace(0, 1, N+1)
X, Y = jnp.meshgrid(x,y)

bcs = [
    jnp.sin(2 * jnp.pi * x), #left boundary
    jnp.sin(2 * jnp.pi * x), #right boundary
    jnp.sin(2 * jnp.pi * y), #lower boundary
    jnp.sin(2 * jnp.pi * y), #upper boundary
]

Id = lx.IdentityLinearOperator(jax.ShapeDtypeStruct((N-1,), jnp.float32))
T = lx.TridiagonalLinearOperator(
    -4 * jnp.ones(N-1),
    jnp.ones(N-2),
    jnp.ones(N-2)
)

def A_mv(u: Float[Array, "(N-1)*(N-1)"]) -> Float[Array, "(N-1)*(N-1)"]:
    chunks = jnp.split(u, N-1)
    A_mv = [
        T.mv(chunks[0]) + chunks[1],
        *(
            chunks[i-1] + T.mv(chunks[i]) + chunks[i+1]
            for i in range(1,N-2)
        ),
        chunks[-2] + T.mv(chunks[-1])
    ]
    return jnp.concat(A_mv)

u_free = jnp.zeros((N-1,N-1))
