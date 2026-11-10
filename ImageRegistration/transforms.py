import numpy as np
import numpy.typing as npt


def _create_transform(tx: float, ty: float, s: float, theta: float, g: float, h: float) -> npt.NDArray[np.float32]:
    """
    Create projective transform matrix with given parameters.
    
    ## Params
    `tx` X translation in pixels.\n
    `ty` Y translation in pixels.\n
    `s` Scale. Must be non-zero positive.\n
    `theta` Rotation angle in radian, counterclockwise.\n
    `g` and `h` are projective parameters. Should be << 1 for 'weak projective' assumption.
    
    ## Return
    NDArray[np.float32] with shape (3, 3):\n
    ┌ s\\*cos_t, -s\\*sin_t,  tx  ┐\n
    │ s\\*sin_t,  s\\*cos_t,  ty  │\n
    └ g,         h,         1   ┘\n
    """
    assert s > 0, "Scale must be non-zero positive"
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)

    matrix = np.array(
        [[s*cos_t, -s*sin_t,  tx],
         [s*sin_t,  s*cos_t,  ty],
         [g,        h,        1]],
         dtype=np.float32
    )
    return matrix


def _get_random_params(
    tx_range: list = [-50, 50], 
    ty_range: list = [-50, 50], 
    s_range: list = [0.9, 1.1], 
    theta_range: list = [-0.1, 0.1], 
    gh_range: list = [0, 0.1]
) -> tuple[float, float, float, float, float, float]:
    """Generate random params for `_create_transform()`"""
    tx = np.random.uniform(tx_range[0], tx_range[1])
    ty = np.random.uniform(ty_range[0], ty_range[1])
    s = np.random.uniform(s_range[0], s_range[1])
    theta = np.random.uniform(theta_range[0], theta_range[1])
    g = np.random.uniform(gh_range[0], gh_range[1])
    h = np.random.uniform(gh_range[0], gh_range[1])
    return tx, ty, s, theta, g, h


if __name__ == "__main__":
    p = np.array([1, 1, 1]).T
    tx, ty, s, theta, g, h = _get_random_params()
    # mat = _create_transform(tx=5, ty=1, s=1, theta=np.pi/2, g=0, h=0)
    mat = _create_transform(tx, ty, s, theta, g, h)
    print(mat @ p)
    print(np.linalg.inv(mat) @ (mat @ p))