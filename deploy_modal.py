from io import BytesIO
from pathlib import Path

from modal import Image, App
import modal

app = App(
    "advanced-liveportrait"
)

CACHE_PATH = "/model_cache/liveportrait"

image = (
    Image.debian_slim(python_version="3.11")
    .apt_install("python3-opencv")
    .add_local_dir(local_path="../model", remote_path=CACHE_PATH, copy=True)
    .add_local_dir(local_path="./", remote_path="/root", copy=True)
    .workdir("/root")
    .pip_install(
        "opencv-python",
        "numpy>=1.26.4",
        "opencv-python-headless",
        "imageio-ffmpeg>=0.5.1",
        "lmdb>=1.4.1",
        "timm>=1.0.7",
        "rich>=13.7.1",
        "albumentations>=1.4.10",
        "ultralytics",
        "tyro==0.8.5",
        "dill",
        "opencv-python",
        "requests",
        "pyaml",
        "torch",
        "safetensors",
        "tqdm"
    )
    #.run_function(load_model)
)

with image.imports():
    import PIL
    
    from test import load_image, ExpressionEditor
    from datetime import datetime
    import base64

@app.cls(
    image=image, gpu="T4", scaledown_window=45, enable_memory_snapshot=True
)
class WebApp:
    @modal.enter(snap=True)
    def startup(self):
        print(f"Downloading models if necessary...")
        self.editor = ExpressionEditor()

    @modal.method()
    def inference(
        self, image_bytes: bytes, **kwargs
    ) -> bytes:
        image = load_image(PIL.Image.open(BytesIO(image_bytes)))
        return self.editor.run(src_image=image, **kwargs)
    
@app.local_entrypoint()
def main(
    image_path=Path(__file__).parent / "reference.jpg",
    output_path=Path("/tmp/stable-diffusion/output.png")
):
    print(f"🎨 reading input image from {image_path}")
    input_image_bytes = Path(image_path).read_bytes()
    output_image_bytes = WebApp().inference.remote(input_image_bytes)

    # if isinstance(output_path, str):
    #     output_path = Path(output_path)

    # dir = output_path.parent
    # dir.mkdir(exist_ok=True, parents=True)

    # print(f"🎨 saving output image to {output_path}")
    # output_path.write_bytes(output_image_bytes)