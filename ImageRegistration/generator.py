# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "marimo>=0.23.3",
# ]
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import numpy.typing as npt
    import cv2
    from pathlib import Path

    return Path, cv2, np, npt


@app.cell
def _(Path):
    DIR_PY_FILE = Path(__file__).resolve().parent
    DIR_SOURCE_IMGS = DIR_PY_FILE / "source_imgs"

    SCAN_WIDTH = 2000
    SCAN_HEIGHT = 6000
    BORDER = 100

    return BORDER, DIR_SOURCE_IMGS, SCAN_HEIGHT, SCAN_WIDTH


@app.function
def generate_scan_rects(sw, sh, b):
    """
    Return centered scan rectangles.\n
    Input: scan_width, scan_height, border.\n
    Output: tuple
    """
    rect1 = (
        (b, b),
        (sw - b, b),
        (b, sh//3 - b),
        (sw - b, sh//3 - b)
    )
    rect2 = (
        (b, sh//3 +b),
        (sw - b, sh//3 + b),
        (b, 2*sh//3 - b),
        (sw - b, 2*sh//3 - b)
    )
    rect3 = (
        (b, 2*sh//3 +b),
        (sw - b, 2*sh//3 + b),
        (b, sh - b),
        (sw - b, sh - b)
    )
    return rect1, rect2, rect3


@app.cell
def _():
    return


@app.cell
def _(np):
    def generate_random_params(
        tx_range: list = [-50, 50], 
        ty_range: list = [-50, 50], 
        s_range: list = [0.7, 0.9], 
        theta_range: list = [-0.1, 0.1], 
        gh_range: list = [0, 0.1]
    ) -> tuple[float, float, float, float, float, float]:
        """Generate random params for `create_transform()`"""
        tx = np.random.uniform(tx_range[0], tx_range[1])
        ty = np.random.uniform(ty_range[0], ty_range[1])
        s = np.random.uniform(s_range[0], s_range[1])
        theta = np.random.uniform(theta_range[0], theta_range[1])
        g = np.random.uniform(gh_range[0], gh_range[1])
        h = np.random.uniform(gh_range[0], gh_range[1])
        return tx, ty, s, theta, g, h

    return (generate_random_params,)


@app.cell
def _(np, npt):
    def create_transform(tx: float, ty: float, s: float, theta: float, g: float, h: float) -> npt.NDArray[np.float32]:
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
        ┌ s\\*cos_t, -s\\*sin_t,  tx  ┐\n
        │ s\\*sin_t,  s\\*cos_t,  ty  │\n
        └ g,         h,         1   ┘\n
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

    return (create_transform,)


@app.cell
def _(cv2, np):
    def apply_transform_to_points(array, H):
        """Require centered points!"""
        _points = np.array(array, dtype=np.float32)
        _points_reshaped = _points.reshape(-1, 1, 2)
        _transformed_points = cv2.perspectiveTransform(_points_reshaped, H)
        _transformed_points = _transformed_points.reshape(-1, 2)
        return _transformed_points

    return (apply_transform_to_points,)


@app.cell
def _():
    return


@app.cell
def _(
    BORDER,
    DIR_SOURCE_IMGS,
    SCAN_HEIGHT,
    SCAN_WIDTH,
    apply_transform_to_points,
    create_transform,
    cv2,
    generate_random_params,
    np,
):
    img_scan = np.ones((SCAN_HEIGHT, SCAN_WIDTH), dtype=np.uint8) * 255
    img_gt = cv2.imread(DIR_SOURCE_IMGS / "1.png")
    IMG_HEIGHT, IMG_WIDTH, _ = img_gt.shape
    rect_scan_array = list(generate_scan_rects(SCAN_WIDTH, SCAN_HEIGHT, BORDER))
    rect_gt_array = [0,0,0]
    H_scan2gt_array = [0,0,0]

    for i in range(3):
        params = generate_random_params(
            tx_range = [-50, 50], 
            ty_range = [-50, 50], 
            s_range = [1, 1], 
            theta_range = [-0.1, 0.1], 
            gh_range = [0, 0.1]
        )
        H_scan2gt_array[i] = create_transform(*params)
        _rect_scan = rect_scan_array[i] - np.array([SCAN_WIDTH/2, SCAN_HEIGHT/6 + i * SCAN_HEIGHT/3])
        _rect_gt = apply_transform_to_points(
            _rect_scan, 
            H_scan2gt_array[i]
        )
        print(_rect_gt)
        rect_gt_array[i] = _rect_gt + np.array([IMG_WIDTH/2, IMG_HEIGHT/2])
        print(np.array([IMG_WIDTH/2, IMG_HEIGHT/2]))
        print(rect_gt_array[i])
        break

        cv2.rectangle(img_scan, rect_scan_array[i][0], rect_scan_array[i][3], 0)
        cv2.rectangle(img_gt, np.int32(rect_gt_array[i][0]), np.int32(rect_gt_array[i][3]), (0,0,255), 20)
        print(rect_gt_array[i])
        print(np.int32(rect_gt_array[i][0]), np.int32(rect_gt_array[i][3]))
        print()

    cv2.imwrite("scan.png", img_scan)
    cv2.imwrite("img.png", img_gt)
    return


if __name__ == "__main__":
    app.run()
