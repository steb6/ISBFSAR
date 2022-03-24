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
        Profile().add("img", min=(1, 3, 480, 640), opt=(1, 3, 480, 640), max=(1, 3, 480, 640))
    ]
    i = {"img": np.ones(shape=(1, 3, 480, 640), dtype=float)}
    create_engine(p,
                  'modules/focus/head_detection/modules/onnx/longest.onnx',
                  'modules/focus/head_detection/modules/engine/longest.engine',
                  i)
