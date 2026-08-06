import cv2
import numpy as np
from typing import Tuple, Optional


class ImageAligner:
    """Aligns completed form image to template reference image using ORB feature matching and Homography."""

    def __init__(self, max_features: int = 2000, keep_percent: float = 0.2):
        self.max_features = max_features
        self.keep_percent = keep_percent
        self.orb = cv2.ORB_create(self.max_features)

    def align_images(
        self, image_np: np.ndarray, template_np: np.ndarray
    ) -> Tuple[np.ndarray, Optional[np.ndarray], float]:
        """
        Aligns image_np to match template_np coordinates.
        Returns:
            aligned_image: Warped aligned image
            homography_matrix: The 3x3 transformation matrix
            match_score: Quality score of keypoint alignment (0.0 to 1.0)
        """
        if len(image_np.shape) == 3:
            gray_img = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        else:
            gray_img = image_np

        if len(template_np.shape) == 3:
            gray_tpl = cv2.cvtColor(template_np, cv2.COLOR_BGR2GRAY)
        else:
            gray_tpl = template_np

        # Detect keypoints and descriptors
        kp1, des1 = self.orb.detectAndCompute(gray_img, None)
        kp2, des2 = self.orb.detectAndCompute(gray_tpl, None)

        if des1 is None or des2 is None or len(des1) < 4 or len(des2) < 4:
            # Fallback to simple resize if keypoints insufficient
            h, w = template_np.shape[:2]
            aligned = cv2.resize(image_np, (w, h))
            return aligned, None, 0.0

        # Match features using Hamming distance
        matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
        matches = list(matcher.match(des1, des2, None))

        # Sort matches by distance (lower is better)
        matches.sort(key=lambda x: x.distance, reverse=False)

        # Retain top matches
        num_good = int(len(matches) * self.keep_percent)
        num_good = max(num_good, 4)
        good_matches = matches[:num_good]

        # Extract location of good matches
        points1 = np.zeros((len(good_matches), 2), dtype=np.float32)
        points2 = np.zeros((len(good_matches), 2), dtype=np.float32)

        for i, match in enumerate(good_matches):
            points1[i, :] = kp1[match.queryIdx].pt
            points2[i, :] = kp2[match.trainIdx].pt

        # Compute Homography matrix
        homography, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

        if homography is None:
            h, w = template_np.shape[:2]
            aligned = cv2.resize(image_np, (w, h))
            return aligned, None, 0.0

        # Warp perspective
        h, w = template_np.shape[:2]
        aligned_image = cv2.warpPerspective(image_np, homography, (w, h))

        # Calculate alignment score based on RANSAC inliers
        inliers = np.sum(mask) if mask is not None else 0
        match_score = float(inliers / max(len(good_matches), 1))

        return aligned_image, homography, round(match_score, 3)
