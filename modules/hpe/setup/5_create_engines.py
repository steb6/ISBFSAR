import numpy as np
from polygraphy.backend.trt import CreateConfig, EngineFromNetwork, NetworkFromOnnxPath, SaveEngine, TrtRunner, Profile, \
    EngineFromBytes

BATCH_SIZE = 5


def create_engine(profiles, in_path, out_path, inputs):
    build_engine = EngineFromNetwork(
        NetworkFromOnnxPath(in_path),
        config=CreateConfig(fp16=True, profiles=profiles, max_workspace_size=10000 << 20)
    )
    build_engine = SaveEngine(build_engine, path=out_path)
    with TrtRunner(build_engine) as runner:
        # NOTE: The runner owns the output buffers and is free to reuse them between `infer()` calls.
        # Thus, if you want to store results from multiple inferences, you should use `copy.deepcopy()`.
        outputs = runner.infer(feed_dict=inputs)

        print("Inference succeeded!")


if __name__ == "__main__":
    # YOLO
    p = [
        Profile().add("input", min=(1, 3, 256, 256), opt=(1, 3, 256, 256), max=(1, 3, 256, 256))
    ]
    i = {"input": np.ones(shape=(1, 3, 256, 256), dtype=int)}
    create_engine(p,
                  'modules/hpe/modules/onnxs/yolo.onnx',
                  'modules/hpe/modules/engines/yolo.engine',
                  i)

    # Image Transformation
    p = [
        Profile().add("frame", min=(480, 640, 3), opt=(480, 640, 3), max=(480, 640, 3)),
        Profile().add("H", min=(5, 3, 3), opt=(5, 3, 3), max=(5, 3, 3))
    ]
    i = {"frame": np.ones(shape=(480, 640, 3), dtype=int),
         "H": np.ones(shape=(5, 3, 3), dtype=np.float32)}
    create_engine(p,
                  'modules/hpe/modules/onnxs/image_transformation.onnx',
                  'modules/hpe/modules/engines/image_transformation.engine',
                  i)

    # BackBone
    p = [
        Profile().add("images", min=(BATCH_SIZE, 256, 256, 3),
                      opt=(BATCH_SIZE, 256, 256, 3),
                      max=(BATCH_SIZE, 256, 256, 3))
    ]
    i = {"images": np.ones(shape=(BATCH_SIZE, 256, 256, 3), dtype=np.float32)}
    create_engine(p,
                  'modules/hpe/modules/onnxs/bbone.onnx',
                  'modules/hpe/modules/engines/bbone.engine',
                  i)



