import argparse

from fluxrt import StreamProcessor
from fluxrt.utils import crop_maximal_rectangle
import cv2
import numpy as np
import threading
import time

MASK_COLOR = np.array([30, 140, 255], dtype=np.uint8)
MASK_ALPHA = 0.45
BRUSH_RADIUS = 15


def main():
    parser = argparse.ArgumentParser(description="Run FluxRT paint demo.")
    parser.add_argument("--int8", action="store_true", help="Enable int8 quantization")
    args = parser.parse_args()

    config_path = "configs/paint_config.json"

    stream_processor = StreamProcessor(config_path)
    input_tensor = stream_processor.get_input_tensor()
    output_tensor = stream_processor.get_output_tensor()

    if args.int8:
        stream_processor.enable_quantization()
    stream_processor.start()

    print("Initializing...")
    while not stream_processor.is_ready():
        time.sleep(0.1)

    resolution = stream_processor.get_resolution()
    h, w = resolution["height"], resolution["width"]
    mask_h, mask_w = h // 16, w // 16

    image = np.ones((h, w, 3), dtype=np.uint8) * 32
    # image = cv2.imread("assets/background.png") # uncomment to load a custom image

    resized_frame = crop_maximal_rectangle(image, h, w)
    input_tensor.copy_from(resized_frame)

    canvas = np.zeros((h, w), np.uint8)

    apply_event = threading.Event()
    prompt_lock = threading.Lock()
    pending_prompt = [None]

    def terminal_thread():
        print("\nType a prompt + Enter to set it.")
        print("Press Enter alone to apply.\n")
        while True:
            try:
                text = input()
            except EOFError:
                break
            if text.strip() == "":
                apply_event.set()
            else:
                with prompt_lock:
                    pending_prompt[0] = text.strip()

    threading.Thread(target=terminal_thread, daemon=True).start()

    win = "FluxRT paint demo"
    cv2.namedWindow(win)

    drawing = [False]
    last_pt = [None]

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing[0] = True
            last_pt[0] = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and drawing[0]:
            cv2.line(canvas, last_pt[0], (x, y), 255, BRUSH_RADIUS * 2)
            last_pt[0] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing[0] = False
            cv2.line(canvas, last_pt[0], (x, y), 255, BRUSH_RADIUS * 2)
        elif event == cv2.EVENT_RBUTTONDOWN:
            canvas[:] = 0

    cv2.setMouseCallback(win, on_mouse)

    while True:
        with prompt_lock:
            if pending_prompt[0] is not None:
                stream_processor.set_prompt(pending_prompt[0])
                print(f"Prompt set: {pending_prompt[0]}")
                pending_prompt[0] = None

        if apply_event.is_set():
            apply_event.clear()
            input_tensor.copy_from(output_tensor.to_numpy())
            canvas[:] = 0
            print("Applied")

        mask_small = cv2.resize(
            canvas, (mask_w, mask_h), interpolation=cv2.INTER_NEAREST
        )
        mask_small = cv2.dilate(mask_small, np.ones((3, 3), np.uint8), iterations=1)
        mask_small = (mask_small > 127).astype(np.uint8)
        stream_processor.set_mask(mask_small * 2)

        frame = output_tensor.to_numpy()

        mask_full = cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_NEAREST)

        overlay = frame.copy()
        overlay[mask_full > 0] = MASK_COLOR
        display = cv2.addWeighted(frame, 1.0 - MASK_ALPHA, overlay, MASK_ALPHA, 0)

        cv2.imshow(win, display)

        key = cv2.waitKey(1000 // 30) & 0xFF
        if key == 27 or key == ord("q"):
            break
        elif key == 13 or key == 10:
            apply_event.set()

    cv2.destroyAllWindows()
    stream_processor.stop()


if __name__ == "__main__":
    main()
