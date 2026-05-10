import argparse

from fluxrt import StreamProcessor
from fluxrt.utils import crop_maximal_rectangle
import cv2
import time


def main():
    parser = argparse.ArgumentParser(description="Run FluxRT cv2 demo.")
    parser.add_argument("--int8", action="store_true", help="Enable int8 quantization")
    args = parser.parse_args()

    config_path = "configs/stream_processor_config.json"

    stream_processor = StreamProcessor(config_path)
    input_tensor = stream_processor.get_input_tensor()
    output_tensor = stream_processor.get_output_tensor()

    if args.int8:
        stream_processor.enable_quantization()
    stream_processor.start()
    stream_processor.set_prompt(
        "Turn this image into cyberpunk night, red and blue neon lamps, cinematic lighting, bokeh"
    )

    resolution = stream_processor.get_resolution()
    cap = cv2.VideoCapture(0)

    print("Initializing...")
    while not stream_processor.is_ready():
        time.sleep(0.1)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        resized_frame = crop_maximal_rectangle(
            frame, resolution["height"], resolution["width"]
        )
        input_tensor.copy_from(resized_frame)

        processed_frame = output_tensor.to_numpy()
        cv2.imshow("Processed Stream", processed_frame)

        if cv2.waitKey(1000 // 25) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    stream_processor.stop()


if __name__ == "__main__":
    main()
