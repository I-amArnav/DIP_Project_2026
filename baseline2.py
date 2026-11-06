from pathlib import Path
import cv2
import numpy as np
from scipy.signal import convolve2d
from skimage.morphology import skeletonize


def get_endpoint_tangent(skeleton, endpoint, path_length=5):
    """
    Traces back from the endpoint along the skeleton to estimate 
    the local stroke direction vector.
    """
    x, y = endpoint
    h, w = skeleton.shape
    current = (x, y)
    visited = {current}
    path = [current]

    for _ in range(path_length):
        cx, cy = current
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < w and 0 <= ny < h and skeleton[ny, nx] == 1:
                    if (nx, ny) not in visited:
                        neighbors.append((nx, ny))
        if not neighbors:
            break
        # Pick the first unvisited neighbor along the skeleton branch
        current = neighbors[0]
        visited.add(current)
        path.append(current)

    if len(path) < 2:
        return None

    # Vector from endpoint into the stroke
    vector = np.array(path[-1]) - np.array(path[0])
    norm = np.linalg.norm(vector)
    if norm == 0:
        return None
    return vector / norm


def baseline_line_completion(input_dir, output_dir, max_bridge_dist=15, max_angle_deviation_deg=45):
    """
    Improved Classical DIP Line Completion using:
    1. Binary thresholding & Skeletonization
    2. Endpoint detection with local tangent/angle validation
    3. Heuristic gap bridging with orientation constraints
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Running Improved Baseline Line Completion on: {input_path.resolve()} ...\n")

    processed_count = 0
    error_count = 0

    kernel = np.array([
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ], dtype=np.uint8)

    max_cos_angle = np.cos(np.radians(max_angle_deviation_deg))

    for file_path in input_path.glob("*.png"):
        try:
            img = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
            lines = (binary < 127).astype(np.uint8)

            skeleton = skeletonize(lines > 0).astype(np.uint8)

            # Distance transform for thickness
            distance_map = cv2.distanceTransform(lines, cv2.DIST_L2, 5)
            skeleton_distances = distance_map[skeleton > 0]
            if len(skeleton_distances) > 0:
                median_radius = np.median(skeleton_distances)
                estimated_thickness = max(1, int(round(2 * median_radius - 1)))
            else:
                estimated_thickness = 1

            # Endpoint Detection
            neighbor_count = convolve2d(skeleton, kernel, mode="same", boundary="fill", fillvalue=0)
            endpoints_mask = (skeleton == 1) & (neighbor_count == 1)
            y_endpoints, x_endpoints = np.where(endpoints_mask)
            endpoints = list(zip(x_endpoints, y_endpoints))

            # Pre-compute tangents for all endpoints
            endpoint_tangents = {}
            for pt in endpoints:
                tan = get_endpoint_tangent(skeleton, pt, path_length=4)
                if tan is not None:
                    endpoint_tangents[pt] = tan

            completed_img = binary.copy()
            connected_pairs = set()

            for i, pt1 in enumerate(endpoints):
                if pt1 not in endpoint_tangents:
                    continue
                for j, pt2 in enumerate(endpoints):
                    if i >= j or pt2 not in endpoint_tangents:
                        continue
                    if (i, j) in connected_pairs or (j, i) in connected_pairs:
                        continue

                    pt1_arr = np.array(pt1)
                    pt2_arr = np.array(pt2)
                    vec_12 = pt2_arr - pt1_arr
                    dist = np.linalg.norm(vec_12)

                    if 3 <= dist <= max_bridge_dist:
                        vec_12_unit = vec_12 / dist
                        tan1 = endpoint_tangents[pt1]
                        tan2 = endpoint_tangents[pt2]

                        # Check orientation: Do the endpoints face each other?
                        # tan1 should point towards pt2 (dot product with vec_12_unit > threshold)
                        # tan2 should point towards pt1 (dot product with -vec_12_unit > threshold)
                        alignment1 = np.dot(tan1, vec_12_unit)
                        alignment2 = np.dot(tan2, -vec_12_unit)

                        if alignment1 >= max_cos_angle and alignment2 >= max_cos_angle:
                            cv2.line(completed_img, pt1, pt2, color=0, thickness=estimated_thickness)
                            connected_pairs.add((i, j))
                            break

            out_file_path = output_path / file_path.name
            cv2.imwrite(str(out_file_path), completed_img)
            processed_count += 1

        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")
            error_count += 1

    print("-" * 40)
    print("Improved Baseline Processing Complete!")
    print(f"Completed images saved: {processed_count}")
    print(f"Output Directory: {output_path.resolve()}")


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline line completion")

    parser.add_argument("input_dir")
    parser.add_argument("output_dir")
    parser.add_argument("--max-bridge-dist", type=int, default=16)
    parser.add_argument("--max-angle-deviation-deg", type=float, default=45)

    args = parser.parse_args()

    baseline_line_completion(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        max_bridge_dist=args.max_bridge_dist,
        max_angle_deviation_deg=args.max_angle_deviation_deg
    )

# Usage:
# python script.py ./Inputs ./Results
# python script.py ./Inputs ./Results --max-bridge-dist 15
# python script.py ./Inputs ./Results --max-bridge-dist 15 --max-angle-deviation-deg 30