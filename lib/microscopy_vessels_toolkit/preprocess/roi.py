import cv2
import numpy as np
from skimage.morphology import binary_closing


class AnnotationNotFound(RuntimeError):
    ...


def load_img_mask(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    mask = img[..., 3] > 125
    img = img[..., :3]
    return img, mask


def extract_tumor_ellipse(img, with_normals=False):
    from skimage.filters import laplace

    b, g, r = split_bgr(img)
    # Extract ellipse edge
    tumor_roi = laplace((r == 255) & ((b + g) < 2)) >= 1

    # Remove smaller objects
    tumor_roi = keep_largest_objects(tumor_roi, 2)

    if not np.any(tumor_roi):
        raise AnnotationNotFound("No red ellipse was found in the image")

    # Match ellipse center and radius
    (cy, cx), (ry, rx) = fast_ellipse_matching(tumor_roi)

    # Compute ellipse pixels coordinates
    tumor_roi_yx = ellipse_yx(cy, cx, ry, rx)

    if not with_normals:
        return tumor_roi_yx
    else:
        center = np.asarray((cy, cx))
        normals = tumor_roi_yx - center[None, :]
        normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
        return tumor_roi_yx, normals


def extract_ROIs(roi_yx, roi_normals, mask, min_h=4, max_h=100, roi_h=250):
    H, W = mask.shape[:2]
    min_yx = np.asarray([[0, 0]], dtype=int)
    max_yx = np.asarray([[H - 1, W - 1]], dtype=int)

    min_probes = np.clip(roi_normals * min_h + roi_yx, min_yx, max_yx).astype(int)
    raw_max_probes = (roi_normals * max_h + roi_yx).astype(int)
    max_probes = np.clip(raw_max_probes, min_yx, max_yx)
    outside_boundaries = np.any(max_probes != raw_max_probes, axis=1)
    in_roi = (mask[min_probes[:, 0], min_probes[:, 1]] | mask[max_probes[:, 0], max_probes[:, 1]]) & ~outside_boundaries

    if np.all(in_roi):
        return [ROI(roi_yx, roi_normals, roi_h)]
    else:
        merge_first_last = in_roi[0] and in_roi[-1]
        in_roi_id = np.arange(len(in_roi))[in_roi]
        in_roi_label = np.cumsum(~in_roi & np.roll(in_roi, 1))[in_roi]
        roi_id = [in_roi_id[in_roi_label == label] for label in range(in_roi_label.max() + 1)]
        if merge_first_last:
            roi_id[0] = np.concatenate([roi_id[-1], roi_id[0]])
            del roi_id[-1]
        return [ROI(roi_yx[_], roi_normals[_], roi_h) for _ in roi_id if len(_) > roi_h]


class ROI:
    def __init__(self, yx, normals, h):
        self.yx = yx
        self.normals = normals
        self.h = h

    def __str__(self):
        return f"{tuple(self.yx[0])} " f"-> {tuple(self.yx[-1])}: {len(self.yx)} px"

    def unwarp(self, img, h=None, resample=1, postprocess=True):
        from scipy.ndimage import map_coordinates

        if h is None:
            h = self.h
        h = np.arange(0, h, 1 / resample)[None, None, :]
        yx = self.yx
        normals = self.normals

        if resample != 1:
            from scipy.interpolate import CubicSpline

            init_t = np.arange(len(yx))
            dest_t = np.arange(0, len(yx) - 1, 1 / resample)

            yx = CubicSpline(init_t, yx, axis=0)(dest_t)
            normals = CubicSpline(init_t, normals, axis=0)(dest_t)

            if resample < 1 and img.dtype == np.uint8:
                img = cv2.GaussianBlur(img, (0, 0), 1 / (resample * 4), img)

        yx = yx.T[..., None]
        normals = normals.T[..., None]

        yx = yx + normals * h
        yx = yx.transpose(0, 2, 1)

        if img.ndim == 3:
            return np.stack([map_coordinates(c, yx) for c in split_bgr(img)], axis=2)
        else:
            m = map_coordinates(img, yx)
            if img.dtype == bool and postprocess:
                m = binary_closing(m, footprint=np.ones((5, 5)))
            return m

    @property
    def upper_yx(self):
        return self.yx + self.normals * self.h

    def mask(self, shape):
        import cv2

        polygon = np.concatenate([self.yx, self.upper_yx[::-1]], axis=0)[:, None, ::-1].astype(np.int32)
        mask = np.zeros(shape, dtype=np.uint8)
        return cv2.fillPoly(mask, [polygon], color=255) > 125

    def draw_polygon(self, img, color=(255, 120, 255), thickness=3, label=None):
        import cv2

        polygon = np.concatenate([self.yx, self.upper_yx[::-1]], axis=0)[:, None, ::-1].astype(np.int32)
        img = cv2.polylines(img, [polygon], True, color, thickness)

        if label:
            mid = len(self.yx) // 2
            y, x = (self.yx[mid] + self.normals[mid] * self.h / 2).astype(int)
            img = cv2.putText(
                img,
                label,
                (x, y),
                cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                thickness / 2,
                color,
                int(thickness * 1.5),
            )


def fast_ellipse_matching(binarized_ellipse):
    y, x = np.where(binarized_ellipse)
    y1, y2 = np.min(y), np.max(y)
    x1, x2 = np.min(x), np.max(x)
    return ((y1 + y2) // 2, ((x1 + x2)) // 2), ((y2 - y1) // 2, (x2 - x1) // 2)


def ellipse_yx(cy, cx, ry, rx):
    theta = np.linspace(0, 2 * np.pi, num=np.ceil(max(ry, rx) * 2 * np.pi).astype(int))
    ell = np.round(np.stack([cy + np.sin(theta) * ry, cx + np.cos(theta) * rx], axis=1)).astype(int)
    duplicates = np.all(ell == np.roll(ell, 1, axis=0), axis=1)
    return np.delete(ell, duplicates, axis=0)


def split_bgr(img):
    return list(img.transpose((2, 0, 1)))


def find_scale(img, mask, return_scale_mask=False, scale_range=750, low_contrast=False):
    from skimage.measure import label, regionprops
    from skimage.filters import apply_hysteresis_threshold

    grey_img = np.mean(img, axis=2)

    if low_contrast:
        scale_mask = apply_hysteresis_threshold(grey_img, 175, 199) | ((grey_img < 1) & mask) & mask
    else:
        scale_mask = apply_hysteresis_threshold(grey_img, 230, 250) | ((grey_img < 1) & mask) & mask

    scale_mask_label = label(scale_mask, connectivity=1)
    filtered_scale_mask = np.zeros_like(scale_mask)

    base_yx = None
    base_length = None
    for region in sorted(regionprops(scale_mask_label), key=lambda x: x.area, reverse=True):
        if is_plain_line(region):
            if base_yx is None:
                base_yx = region.centroid
                base_length = region.major_axis_length
            elif (
                np.abs(base_yx[0] - region.centroid[0]) > 10
                or np.abs(base_yx[1] - region.centroid[1]) > base_length * 2
            ):
                continue
            filtered_scale_mask[region.label == scale_mask_label] = True

    scale_idx = np.where(filtered_scale_mask)
    if len(scale_idx[0]) == 0:
        if not low_contrast:
            print("DETECTING SCALE WITH LOW CONTRAST")
            return find_scale(
                img,
                mask,
                return_scale_mask=return_scale_mask,
                scale_range=scale_range,
                low_contrast=True,
            )

        if not return_scale_mask:
            raise AnnotationNotFound("Scale was not found")
        else:
            return np.nan, scale_mask
    x0 = scale_idx[1].min()
    x1 = scale_idx[1].max()

    scale = scale_range / (x1 - x0)

    if return_scale_mask:
        return scale, filtered_scale_mask
    else:
        return scale


def is_plain_line(region):
    y0, x0, y1, x1 = region.bbox
    h, w = y1 - y0 + 1, x1 - x0 + 1

    def to_rad(deg):
        return deg * np.pi / 180

    # if(w>40 and h<20):
    #    print(f'shape: {(h, w)};\t solidity: {region.solidity:.2f};'
    #           '\t orientation: {rad2deg(region.orientation):.0f};'
    #           '\t minor_axis: {region.axis_minor_length:.2f}')

    return (
        (w > 3 * h and 2 <= h < 20 and w > 40)
        and region.solidity > 0.8
        and -5 < rad2deg(region.orientation) < 5
        and 1.5 <= region.axis_minor_length < 9
    )


def rad2deg(orientation, shift=90):
    deg = orientation * 180 / np.pi - shift
    return (deg + 90) % 180 - 90


def keep_largest_objects(binary_map, top=1):
    from skimage.measure import label

    labels = label(binary_map)
    idx = np.argsort(np.bincount(labels.flatten())[1:])[-top:] + 1
    return np.isin(labels, idx)


def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        raise Exception("lines do not intersect")

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y
