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
    from copy import deepcopy
    import json

    return Path, cv2, deepcopy, json, np, npt


@app.cell
def _(Path):
    DIR_PY_FILE = Path(__file__).resolve().parent
    DIR_SOURCE_IMGS = DIR_PY_FILE / "source_imgs"

    SCAN_WIDTH = 2000
    SCAN_HEIGHT = 6000
    BORDER = 100
    BACKGROUND_COLOR = 0
    CHANNEL_NAMES = ("B", "G", "R")
    return (
        BACKGROUND_COLOR,
        BORDER,
        CHANNEL_NAMES,
        DIR_SOURCE_IMGS,
        SCAN_HEIGHT,
        SCAN_WIDTH,
    )


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
        gh_range: list = [-0.00005, 0.00005]
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
def _(CHANNEL_NAMES, Path, cv2, json):
    CORNER_ORDER = ["top_left", "top_right", "bottom_right", "bottom_left"]

    def save_data(
        img_gt,
        img_scan,
        rect_scan_array,
        rect_gt_array,
        model: str,
        original_image: str,
        res_path: str
    ):
        """
        Save dataset unit: `img_gt.png`, `scan.png`, `config.json`.\n
        Points of `rect_scan_array` and `rect_gt_array` are ordered as
        `top_left`, `top_right`, `bottom_left`, `bottom_right` and
        reordered on save to `CORNER_ORDER`.
        """
        def _reorder(pts):
            return [pts[0], pts[1], pts[3], pts[2]]

        IMG_HEIGHT, IMG_WIDTH, _ = img_gt.shape
        SCAN_HEIGHT, SCAN_WIDTH = img_scan.shape

        _transforms = []
        for _name, _rect_scan, _quad_gt in zip(CHANNEL_NAMES, rect_scan_array, rect_gt_array):
            _transforms.append(
                {
                    "channel": _name,
                    "scan_quad": [[int(x), int(y)] for x, y in _reorder(_rect_scan)],
                    "original_quad": [[round(float(x), 6), round(float(y), 6)] for x, y in _reorder(_quad_gt)]
                }
            )

        _config = {
            "model": model,
            "corner_order": CORNER_ORDER,
            "original_size": {"width": IMG_WIDTH, "height": IMG_HEIGHT},
            "scan_size": {"width": SCAN_WIDTH, "height": SCAN_HEIGHT},
            "transforms": _transforms,
            "original_image": original_image
        }

        Path(res_path).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(Path(res_path) / "img_gt.png", img_gt)
        cv2.imwrite(Path(res_path) / "scan.png", img_scan)
        (Path(res_path) / "config.json").write_text(
            json.dumps(_config, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

    return (save_data,)


@app.cell
def _(
    BACKGROUND_COLOR,
    BORDER,
    DIR_SOURCE_IMGS,
    Path,
    SCAN_HEIGHT,
    SCAN_WIDTH,
    apply_transform_to_points,
    create_transform,
    cv2,
    deepcopy,
    generate_random_params,
    np,
    save_data,
):
    img_scan = np.ones((SCAN_HEIGHT, SCAN_WIDTH), dtype=np.uint8) * BACKGROUND_COLOR
    img_gt = cv2.imread(DIR_SOURCE_IMGS / "1.png")
    img_gt_rects = deepcopy(img_gt)
    IMG_HEIGHT, IMG_WIDTH, _ = img_gt.shape
    rect_scan_array = list(generate_scan_rects(SCAN_WIDTH, SCAN_HEIGHT, BORDER))
    rect_gt_array = [0,0,0]
    H_scan2gt_array = [0,0,0]

    for i in range(3):
        params = generate_random_params(
            tx_range = [-50, 50], 
            ty_range = [-50, 50], 
            s_range = [0.50, 0.60], 
            theta_range = [-0.1, 0.1], 
            gh_range = [-0.00005, 0.00005]
        )
        H_scan2gt_array[i] = create_transform(*params)
        _rect_scan = rect_scan_array[i] - np.array([SCAN_WIDTH/2, SCAN_HEIGHT/6 + i * SCAN_HEIGHT/3])
        _rect_gt = apply_transform_to_points(
            _rect_scan, 
            H_scan2gt_array[i]
        )
        rect_gt_array[i] = _rect_gt + np.array([IMG_WIDTH/2, IMG_HEIGHT/2])

        _rect_x0, _rect_y0 = rect_scan_array[i][0]
        _rect_x1, _rect_y1 = rect_scan_array[i][3]
        _T_img2center = np.array(
            [[1, 0, -IMG_WIDTH/2],
             [0, 1, -IMG_HEIGHT/2],
             [0, 0, 1]],
            dtype=np.float32
        )
        _T_center2rect = np.array(
            [[1, 0, (_rect_x1 - _rect_x0)/2],
             [0, 1, (_rect_y1 - _rect_y0)/2],
             [0, 0, 1]],
            dtype=np.float32
        )
        _H_inv = np.linalg.inv(H_scan2gt_array[i])
        _H_gt2scan = _T_center2rect @ _H_inv @ _T_img2center
        _img_channel = cv2.warpPerspective(
            img_gt,
            _H_gt2scan,
            (_rect_x1 - _rect_x0, _rect_y1 - _rect_y0)
        )
        img_scan[_rect_y0:_rect_y1, _rect_x0:_rect_x1] = _img_channel[:,:,i]


        pts = np.int32([rect_gt_array[i][0], rect_gt_array[i][1], rect_gt_array[i][3], rect_gt_array[i][2]])
        cv2.polylines(
            img_gt_rects,
            [pts.reshape((-1, 1, 2))],
            isClosed=True,
            color=(255 * (i==0), 255 * (i==1), 255 * (i==2)),
            thickness=3,
        )

    save_data(
        img_gt, 
        img_scan, 
        rect_scan_array, 
        rect_gt_array,
        model = "Слабопроективное",
        original_image = "source_imgs/1.png",
        res_path = Path("gen_data") / "1"
        )
    cv2.imwrite("scan_rects.png", img_scan)
    cv2.imwrite("img_rects.png", img_gt_rects)
    return


if __name__ == "__main__":
    app.run()
