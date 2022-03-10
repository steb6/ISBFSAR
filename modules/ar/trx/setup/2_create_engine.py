import numpy as np
from polygraphy.backend.trt import CreateConfig, EngineFromNetwork, NetworkFromOnnxPath, SaveEngine, TrtRunner, Profile


BATCH_SIZE = 1


def create_engine(profiles, in_path, out_path, inputs):
    build_engine = EngineFromNetwork(
        NetworkFromOnnxPath(in_path),
        config=CreateConfig(fp16=True, max_workspace_size=10000 << 20)  # profiles=profiles,
    )
    build_engine = SaveEngine(build_engine, path=out_path)

    with TrtRunner(build_engine) as runner:
        # NOTE: The runner owns the output buffers and is free to reuse them between `infer()` calls.
        # Thus, if you want to store results from multiple inferences, you should use `copy.deepcopy()`.
        outputs = runner.infer(feed_dict=inputs)

        print("Inference succeeded!")


if __name__ == "__main__":
    # TRX USA ROSASCO METHOD
    p = [
        Profile().add("support", min=(80, 90), opt=(80, 90), max=(80, 90)),
        Profile().add("query", min=(16, 90), opt=(16, 90), max=(16, 90)),
        Profile().add("labels", min=(5), opt=(5), max=(5))
    ]
    i = {"image": np.ones(shape=(80, 90), dtype=float),
         "query": np.ones(shape=(16, 90), dtype=float),
         "labels": np.ones(shape=(5), dtype=int)}
    create_engine(p,
                  'modules/ar/trx/checkpoints/trx.onnx',
                  'modules/ar/trx/checkpoints/trx.engine',
                  i)

