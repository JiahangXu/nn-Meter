# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import jsonlines
import nn_meter
from nn_meter.dataset import bench_dataset

def create_dummy_input(name):
    dummy_input = {
        "inbounds": [],
        "attr": {
            "name": name,
            "type": "Placeholder",
            "output_shape": [],
            "attr": {},
            "input_shape": []
        },
        "outbounds": []
    }
    return dummy_input

predictors = nn_meter.list_latency_predictors()
for p in predictors:
    print(f"[Predictor] {p['name']}: version={p['version']}")
    # load predictor
    predictor_name = 'adreno640gpu_tflite21'
    predictor_version = 1.0
    predictor = nn_meter.load_latency_predictor(predictor_name, predictor_version)

    datasets = bench_dataset()
    test_data = datasets[0]
    print(datasets)

    with jsonlines.open(test_data) as data_reader:
        n = len(data_reader)
        for i, item in enumerate(data_reader):
            print(f'{i}/{n}')
            model = item['graph']

            for node_name, node in model.items():
                if node["inbounds"] == []:
                    continue
                dummy_model = {}
                for input in node["inbounds"]:
                    dummy_model[input] = create_dummy_input(input)
                dummy_model[node_name] = node
                latency = predictor.predict(dummy_model, model_type="nnmeter-ir")
                if "latency" not in node["attr"]:
                    node["attr"]["latency"] = {}
                node["attr"]["latency"][predictor_name] = latency
            
            item['graph'] = model

            with jsonlines.open('output.jsonl', mode='a') as writer:
                writer.write(item)
