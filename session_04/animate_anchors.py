"""
## Requirements:

numpy
imageio
imageio-ffmpeg
opencv-python(-headless)

## Generating the videos for the presentation

Large anchors and detection boxes:

```
python animate_anchors.py 300 100 --fps 8 --stack --ratios 0.5,1,2
```

Small anchors and detection boxes:

```
python animate_anchors.py 150 50 --fps 32 --stack --ratios 0.5,1,2
```

Ground truth boxes, large anchors, positive anchors with IoU:

```
python animate_anchors.py 300 100 --fps 8 --stack --ratios 0.5,1,2 \
    --gt --iou --video ground_truth.mp4
```
"""
from pathlib import Path
import itertools
import argparse

import numpy as np
import cv2
import imageio


# Ground truth boxes
ticks_gt = np.array([
        [645,154,916,460],
        [437,143,606,347],
        [296,204,393,308],
        [195,222,272,285]
])

RED = (240, 0, 0)
GREEN = (0, 240, 0)
BLUE = (0, 0, 240)


def area(a):
    a[2] = max(a[2], a[0])
    a[3] = max(a[3], a[1])
    return (a[2] - a[0]) * (a[3] - a[1])


def iou(a, b):
    u = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
    i = [max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])]
    return area(i) / max(1, area(u))


def gen_boxes(image, ratios, size, stride):
    for y in np.arange(size//2, image.shape[0], stride):
        for x in np.arange(size//2, image.shape[1], stride):
            for r in ratios:
                b = np.sqrt([r, 1/r]) * size / 2
                b = np.concatenate([-b, b])
                b[0::2] += x
                b[1::2] += y
                yield b


def draw_box(image, box, color, thickness=None, inplace=True):
    thickness = thickness or 2
    if not inplace:
        image = image.copy()
    b = box.astype(int)
    cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), color, thickness, cv2.LINE_AA)
    return image


def draw_text(image, origin, text, color, font_size=12):
    origin = tuple(int(x) for x in origin)
    cv2.putText(
            image, text, origin, cv2.FONT_HERSHEY_PLAIN, 
            fontScale=font_size / 12.0, color=color
    )


def main():
    args = parse_args()
    ratios = [float(x) for x in args.ratios.split(',')]

    image = cv2.imread(args.image)
    image = image[:, :, ::-1]  # BGR -> RGB

    # Create output video
    writer = imageio.get_writer(
            args.video or f'ticks_{args.size}_{args.stride}_{len(ratios)}.mp4', 
            fps=args.fps
    )

    iou_origin = np.array([0, image.shape[0]])

    # Number of images to tile horizontally
    n_copies = len(ratios) if args.stack else 1
    # Tiling index
    box_idx = 0
    # Tiled images
    canvases = []
    # Positive (blue) boxes for each tiled image
    positive_anchors = [[] for _ in range(n_copies)]
    for box in itertools.chain(
            gen_boxes(image, ratios, args.size, args.stride), 
            [None, None, None]  # last frame has no anchor boxes
    ):
        canvas = image.copy()

        # If we're not on the last frame...
        if box is not None:
            # Pick the ground truth box having best IoU with the anchor.
            score = max(iou(box, t) for t in ticks_gt)
            if args.iou:
                draw_text(canvas, iou_origin, f'IoU={score:.2f}', 
                          font_size=18, color=RED)
            if score >= 0.5:
                # This is a positive anchor.
                positive_anchors[box_idx].append((score, box))

        # Draw positive anchors.
        for s, b in positive_anchors[box_idx]:
            draw_box(canvas, b, BLUE)
            if args.iou:
               draw_text(canvas, b[:2], f'{s:.2f}', color=BLUE)

        # If it's not positive, draw a red anchor box.
        if box is not None and score < 0.5:
            draw_box(canvas, box, RED)

        if args.gt:
            # Draw ground truth boxes.
            for t in ticks_gt:
                draw_box(canvas, t, GREEN)

        canvases.append(canvas)
        box_idx = (box_idx + 1) % n_copies

        if box_idx == 0:
            full_canvas = np.concatenate(canvases, axis=1)
            writer.append_data(full_canvas)
            canvases = []

    writer.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Apply model to a video.")
    parser.add_argument("size", type=int, help="Base size of anchors in pixels.")
    parser.add_argument("stride", type=int, help="Anchor stride in pixels.")
    parser.add_argument("--ratios", default="1", 
                        help="Anchor aspect ratios; comma-separated values, no spaces. "
                             "Ex: 0.5,1,2" )
    parser.add_argument("--video", help="Custom name for output video.")
    parser.add_argument("--image", default="ticks.png", help="Input image.")
    parser.add_argument("--fps", type=int, default=8, help="Frames per second.")
    parser.add_argument("--iou", action='store_true', help="Draw IoU values.")
    parser.add_argument("--gt", action='store_true', help="Draw groundtruth boxes.")
    parser.add_argument("--stack", action='store_true', 
                        help="Render different aspect ratios side-by-side.")

    return parser.parse_args()


if __name__ == "__main__":
    main()


