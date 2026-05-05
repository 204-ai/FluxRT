from fluxrt import StreamProcessor
from fluxrt.utils import crop_maximal_rectangle
import cv2
import time


def main():
    config_path = "configs/stream_processor_config.json"

    stream_processor = StreamProcessor(config_path)
    input_tensor = stream_processor.get_input_tensor()
    output_tensor = stream_processor.get_output_tensor()

    stream_processor.start()
    stream_processor.set_prompt("Make white background")

    resolution = stream_processor.get_resolution()
    cap = cv2.VideoCapture("video.mp4")

    fps = 25
    output_width = resolution["width"]
    output_height = resolution["height"]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_path = "processed_video.mp4"
    out = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))
    time.sleep(10)

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

        cv2.imshow("Processed Stream", processed_frame)

        if cv2.waitKey(1000 // fps) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    stream_processor.stop()


if __name__ == "__main__":
    main()
