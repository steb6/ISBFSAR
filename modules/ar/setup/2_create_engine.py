import numpy as np
from polygraphy.backend.trt import CreateConfig, EngineFromNetwork, NetworkFromOnnxPath, SaveEngine, TrtRunner, Profile


def create_engine(in_path, out_path, inputs):
    build_engine = EngineFromNetwork(
        NetworkFromOnnxPath(in_path),
        config=CreateConfig(fp16=True, max_workspace_size=7 << 20)  # profiles=profiles,
    )
    build_engine = SaveEngine(build_engine, path=out_path)

    with TrtRunner(build_engine) as runner:
        # NOTE: The runner owns the output buffers and is free to reuse them between `infer()` calls.
        # Thus, if you want to store results from multiple inferences, you should use `copy.deepcopy()`.
        outputs = runner.infer(feed_dict=inputs)

        print("Inference succeeded!")


if __name__ == "__main__":
    i = {"support": np.ones(shape=(80, 90), dtype=np.float32),
         "query": np.ones(shape=(16, 90), dtype=np.float32),
         "labels": np.ones(shape=(5), dtype=np.int32)}
    create_engine('modules/ar/modules/onnxs/FULL.onnx',
                  'modules/ar/modules/engines/trx.engine',
                  i)
