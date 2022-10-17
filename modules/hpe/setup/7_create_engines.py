import numpy as np
from polygraphy.backend.trt import CreateConfig, EngineFromNetwork, NetworkFromOnnxPath, SaveEngine, TrtRunner

BATCH_SIZE = 1


def create_engine(in_path, out_path, inputs):
    build_engine = EngineFromNetwork(
        NetworkFromOnnxPath(in_path),
        config=CreateConfig(fp16=True, max_workspace_size=10000 << 20)  # ,profiles=profiles
    )
    build_engine = SaveEngine(build_engine, path=out_path)
    with TrtRunner(build_engine) as runner:
        # NOTE: The runner owns the output buffers and is free to reuse them between `infer()` calls.
        # Thus, if you want to store results from multiple inferences, you should use `copy.deepcopy()`.
        outputs = runner.infer(feed_dict=inputs)

        print("Inference succeeded!")


if __name__ == "__main__":
    # YOLO
    i = {"input": np.ones(shape=(1, 3, 256, 256), dtype=np.float32)}
    create_engine(  # p,
        'modules/hpe/weights/onnxs/yolo.onnx',
        'modules/hpe/weights/engines/docker/yolo.engine',
        i)

    # Image Transformation
    i = {"frame": np.ones(shape=(480, 640, 3), dtype=np.int32),
         "H": np.ones(shape=(BATCH_SIZE, 3, 3), dtype=np.float32)}
    create_engine(  # p,
        'modules/hpe/weights/onnxs/image_transformation{}.onnx'.format(BATCH_SIZE),
        'modules/hpe/weights/engines/docker/image_transformation{}.engine'.format(BATCH_SIZE),
        i)

    # BackBone
    i = {"images": np.ones(shape=(BATCH_SIZE, 256, 256, 3), dtype=np.float32)}
    create_engine(  # p,
        'modules/hpe/weights/onnxs/bbone{}.onnx'.format(BATCH_SIZE),
        'modules/hpe/weights/engines/docker/bbone{}.engine'.format(BATCH_SIZE),
        i)
    # Heads
    i = {"input": np.ones(shape=(81920*BATCH_SIZE,), dtype=np.float32)}
    create_engine(  # p,
        'modules/hpe/weights/onnxs/heads{}.onnx'.format(BATCH_SIZE),
        'modules/hpe/weights/engines/docker/heads{}.engine'.format(BATCH_SIZE),
        i)
