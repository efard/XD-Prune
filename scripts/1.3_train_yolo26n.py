from ultralytics import YOLO

model = YOLO("yolo26n.pt")

'''
results = model.train(
    data="data/mio_tcd_yolo/mio_tcd.yaml",
    epochs=3,
    imgsz=640,
    batch=8,
    device="cpu",
    workers=2,
    project="runs/mio_tcd_yolo26n",
    name="sanity_check",
)

def main():
    # quick pipeline sanity check:
    # 1. Check data.yaml path
    # 2. Check image-label matching
    # 3. Check class IDs
    # 4. Check whether Ultralytics can start training correctly
    model.train(
        data="data/mio_tcd_yolo/mio_tcd.yaml",
        # an epoch means one full pass over the selected dataset portion
        epochs=1,
        # Use only 1% of the training dataset
        fraction=0.01,
        # Smaller image size makes training faster
        imgsz=320,
        # Small batch size is safer for CPU / low-memory machines
        batch=4,
        # Windows + AMD GPU usually cannot use CUDA training
        device="cpu",
        # Keep workers low on Windows to reduce multiprocessing problems
        workers=2,
        # validation can also scan many images and make the sanity check slow
        val=False,
        # plots are useful later
        plots=False,

        project="runs/mio_tcd_yolo26n",
        name="v2 baseline",
        exist_ok=True,
    )
'''

def main():
    model = YOLO("yolo26n.pt")

    model.train(
        data="data/mio_tcd_yolo/mio_tcd.yaml",

        fraction=0.05,
        epochs=10,
        imgsz=416,
        batch=4,

        device="cpu",
        optimizer="auto",
        val=True,
        plots=True,
        seed=42,
        deterministic=True,
        workers=2,

        project="runs/mio_tcd_yolo26n",
        name="v2_cpu_baseline_5p_10e_416",
        exist_ok=False,
    )

# Required on Windows when running Ultralytics training inside a Python script
# Reason: Windows multiprocessing needs the script entry point to be protected
if __name__ == "__main__":
    main()

'''
python scripts/1.3_train_yolo26n.py
'''