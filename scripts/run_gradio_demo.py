import argparse
import threading
import time

import cv2
import gradio as gr

from fluxrt import StreamProcessor
from fluxrt.utils import crop_maximal_rectangle

default_prompt = "Turn this image into cyberpunk night, red and blue neon lamps, bokeh"

stream_processor = None
input_tensor = None
output_tensor = None
resolution = None
use_int8 = False

stop_video_event = threading.Event()
processor_lock = threading.Lock()
current_video_id = 0
current_video_id_lock = threading.Lock()


def get_processor():
    global stream_processor, input_tensor, output_tensor, resolution

    if stream_processor is None:
        # stream_processor = StreamProcessor("configs/stream_processor_config.json") # uncomment if you dont need reference image
        stream_processor = StreamProcessor("configs/config_with_reference.json")
        if use_int8:
            stream_processor.enable_quantization()
        stream_processor.start()
        stream_processor.set_prompt(default_prompt)

        input_tensor = stream_processor.get_input_tensor()
        output_tensor = stream_processor.get_output_tensor()
        resolution = stream_processor.get_resolution()

    return stream_processor, input_tensor, output_tensor, resolution


def to_bgr(frame):
    if frame is None:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def to_rgb(frame):
    if frame is None:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def process_frame(frame):
    _, input_tensor, output_tensor, resolution = get_processor()
    frame = crop_maximal_rectangle(frame, resolution["height"], resolution["width"])

    with processor_lock:
        input_tensor.copy_from(frame)
        processed = output_tensor.to_numpy()
    return frame, processed


def set_prompt(prompt: str):
    sp, _, _, _ = get_processor()
    sp.set_prompt(prompt)


def set_reference_image_ui(image):
    sp, _, _, _ = get_processor()
    sp.set_reference_image(image)


def switch_mode(mode: str, request: gr.Request | None):
    if mode == "webcam":
        stop_video_event.set()
    elif mode == "local":
        stop_video_event.clear()

    webcam_visible = mode == "webcam"
    local_visible = mode == "local"
    return (
        gr.update(visible=webcam_visible),  # webcam_output_col
        gr.update(visible=local_visible),  # local_output_col
        gr.update(visible=webcam_visible),  # webcam_input_col
        gr.update(visible=local_visible),  # local_input_col
    )


def process_webcam(frame):
    if frame is None:
        return None

    _, processed = process_frame(to_bgr(frame))
    return to_rgb(processed)


def process_local_video(video_path: str | None, request: gr.Request | None):
    global current_video_id
    if not video_path:
        return None, None

    with current_video_id_lock:
        current_video_id += 1
        my_id = current_video_id

    stop_video_event.clear()
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_time = 1.0 / fps

    try:
        while not stop_video_event.is_set():
            with current_video_id_lock:
                if current_video_id != my_id:
                    break
            ok, frame = cap.read()
            if not ok:
                break

            start = time.time()
            input_frame, processed = process_frame(frame)
            yield to_rgb(input_frame), to_rgb(processed)

            elapsed = time.time() - start
            sleep_time = max(0, frame_time - elapsed)
            time.sleep(sleep_time)
    finally:
        cap.release()


def main():
    global use_int8
    parser = argparse.ArgumentParser(description="Run FluxRT Gradio demo.")
    parser.add_argument("--int8", action="store_true", help="Enable int8 quantization")
    args, _ = parser.parse_known_args()
    use_int8 = args.int8

    get_processor()
    use_reference_image = stream_processor.config.get("use_reference_image", False)

    with gr.Blocks() as demo:
        mode = gr.Radio(
            choices=["webcam", "local"],
            value="webcam",
            label="Mode",
        )

        # Top: full-width processed output
        with gr.Column(visible=True) as webcam_output_col:
            webcam_output = gr.Image(
                streaming=True,
                label="Processed stream",
            )

        with gr.Column(visible=False) as local_output_col:
            local_output = gr.Image(streaming=True, label="Processed stream")

        # Bottom row: input | prompt | reference image
        with gr.Row():
            with gr.Column(visible=True) as webcam_input_col:
                webcam_input = gr.Image(
                    sources=["webcam"],
                    streaming=True,
                    type="numpy",
                    label="Webcam",
                )

            with gr.Column(visible=False) as local_input_col:
                video_file = gr.File(
                    label="Choose local video",
                    file_count="single",
                    file_types=["video"],
                    type="filepath",
                )
                local_input = gr.Image(label="Input stream")

            prompt = gr.Textbox(
                value=default_prompt,
                label="Prompt",
                lines=3,
            )

            if use_reference_image:
                ref_image_input = gr.Image(
                    label="Reference Image",
                    type="numpy",
                    sources=["upload"],
                    image_mode="RGB",
                )

        mode.change(
            switch_mode,
            inputs=mode,
            outputs=[
                webcam_output_col,
                local_output_col,
                webcam_input_col,
                local_input_col,
            ],
        )

        webcam_input.stream(
            process_webcam,
            inputs=webcam_input,
            outputs=[webcam_output],
            stream_every=0.04,
            concurrency_limit=1,
        )

        video_file.change(
            process_local_video,
            inputs=video_file,
            outputs=[local_input, local_output],
        )

        prompt.change(set_prompt, inputs=prompt, outputs=None)

        if use_reference_image:
            ref_image_input.change(
                set_reference_image_ui,
                inputs=ref_image_input,
                outputs=None,
            )

    demo.queue(default_concurrency_limit=1).launch()


if __name__ == "__main__":
    main()
