# modal/train_modal.py
from modal import App, Image, Volume, gpu

app = App("bert4rec-train")

image = (
    Image.debian_slim()
    .pip_install(
        "torch==2.3.1",
        "polars[gcsfs]==1.5.0",
    )
)

# Your volume is named "datasets"; data lives at /modelling_data inside it
vol = Volume.from_name("datasets", create_if_missing=False)

@app.function(
    image=image,
    gpu=gpu.L4(),                   # change to A10G/A100 if you prefer
    volumes={"/data": vol},         # volume mounted at /data
    timeout=60 * 60 * 4,            # 4 hours
)
def train_job(
    train_glob: str = "/data/modelling_data/*.parquet",
    out_dir: str = "/data/out",
    epochs: int = 10,
    batch_size: int = 256,
):
    # Import your trainer (ensure trainer/bert4rec_modelling.py exposes train())
    from trainer.bert4rec_modelling import train
    train(
        train_uri=train_glob,
        val_uri=None,
        out_dir=out_dir,
        epochs=epochs,
        batch_size=batch_size,
    )

@app.local_entrypoint()
def main(
    train_glob: str = "/data/modelling_data/*.parquet",
    out_dir: str = "/data/out",
    epochs: int = 10,
    batch_size: int = 256,
):
    train_job.call(train_glob, out_dir, epochs, batch_size)
