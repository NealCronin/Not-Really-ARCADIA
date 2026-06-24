import numpy as np
import cv2

class Heatmap:
    def __init__(self):
        # Kept for backwards compatibility with other modules, but no longer used in the math loop
        self.heat_gamma = 1.0 

    def _create_heatmap(self, image, regions):
        # 1. Initialize a blank floating-point canvas for score accumulation
        heatmap = np.zeros(image.shape[:2], dtype=np.float32)

        if not regions: 
            return image

        for region in regions:
            mask = region.mask.astype(np.float32)
            score = region.score  # Range: 0.0 to 1.0

            # Retain the highest score at any given overlapping pixel location
            heatmap = np.maximum(
                heatmap,
                mask * score
            )

        # 2. Smoothly bleed the values outward to create a natural thermal gradient.
        # (151, 151) provides a cleaner, sharper gradient for aerial drone footage than (301, 301)
        spread = (151, 151)
        sigma = 0
        heatmap = cv2.GaussianBlur(heatmap, spread, sigma)

        # 3. Scale the 0.0-1.0 float values directly to 0-255 8-bit integers
        heatmap = np.clip(heatmap * 255.0, 0, 255).astype(np.uint8)

        # 4. Map the intensities to the JET thermal spectrum (0 = Deep Blue, 255 = Crimson Red)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # 5. Blend the translucent thermal layer over your base drone image
        output = cv2.addWeighted(
            image,    # Base drone image
            0.6,      # Transparency weight of base image
            heatmap,  # Heatmap overlay color image
            0.4,      # Transparency weight of heatmap
            0         # Constant brightness offset
        )

        return output

    def draw_heatmap(self, image, segmentations):
        return self._create_heatmap(image, segmentations)