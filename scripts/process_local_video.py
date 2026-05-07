import argparse

from fluxrt import StreamProcessor
from fluxrt.utils import crop_maximal_rectangle
import cv2
import time
from tqdm import tqdm

DEFAULT_PROMPT = "Turn this image into cyberpunk night, red and blue neon lamps, cinematic lighting, bokeh"


def main():
    parser = argparse.ArgumentParser(description="Process a video with FluxRT.")
    parser.add_argument(
        "--input",
        default="input.mp4",
        help="Path to the input video (default: input.mp4)",
    )
    parser.add_argument(
        "--output",
        default="output.mp4",
        help="Path to the output video (default: output.mp4)",
    )
    parser.add_argument(
        "--prompt", default=DEFAULT_PROMPT, help="Style prompt for processing"
    )
    args = parser.parse_args()

    config_path = "configs/stream_processor_config.json"

    stream_processor = StreamProcessor(config_path)
    input_tensor = stream_processor.get_input_tensor()
    output_tensor = stream_processor.get_output_tensor()

    stream_processor.start()
    stream_processor.set_prompt(args.prompt)

    resolution = stream_processor.get_resolution()
    cap = cv2.VideoCapture(args.input)

    fps = 25
    output_width = resolution["width"]
    output_height = resolution["height"]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.output, fourcc, fps, (output_width, output_height))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress = tqdm(total=total_frames if total_frames > 0 else None, unit="frame")

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

        out.write(processed_frame)
        progress.update(1)

        cv2.imshow("Processed Stream", processed_frame)

        if cv2.waitKey(1000 // fps) & 0xFF == ord("q"):
            break

    progress.close()
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    stream_processor.stop()


if __name__ == "__main__":
    main()
