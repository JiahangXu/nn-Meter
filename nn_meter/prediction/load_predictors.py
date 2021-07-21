import pickle
import os
from glob import glob
from zipfile import ZipFile
from tqdm import tqdm
import requests


def loading_to_local(pred_info, dir="data/predictorzoo"):
    """
    @params:

    configs: the default predictor.yaml that describes the supported hardware+backend
    hardware: the targeting hardware_inferenceframework name
    dir: the local directory to store the kernel predictors and fusion rules

    """
    os.makedirs(dir, exist_ok=True)
    hardware = pred_info['name']
    ppath = os.path.join(dir, hardware)

    isdownloaded = check_predictors(ppath, pred_info["kernel_predictors"])
    if not isdownloaded:
        download_from_url(pred_info["download"], dir)

    # load predictors
    predictors = {}
    ps = glob(os.path.join(ppath, "**.pkl"))
    for p in ps:
        pname =  os.path.basename(p).replace(".pkl", "")
        with open(p, "rb") as f:
            print("load predictor", p)
            model = pickle.load(f)
            predictors[pname] = model
    fusionrule = os.path.join(ppath, "rule_" + hardware + ".json")
    print(fusionrule)
    if not os.path.isfile(fusionrule):
        raise ValueError(
            "check your fusion rule path, file " + fusionrule + " does not exist！"
        )
    return predictors, fusionrule


def download_from_url(urladdr, ppath):
    """
    download the kernel predictors from the url
    @params:

    urladdr: github release url address
    ppath: the targeting hardware_inferenceframework name

    """
    file_name = os.path.join(ppath, ".zip")
    if not os.path.isdir(ppath):
        os.makedirs(ppath)

    print("download from " + urladdr)
    response = requests.get(urladdr, stream=True)
    total_size_in_bytes = int(response.headers.get("content-length", 0))
    block_size = 2048  # 2 Kibibyte
    progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)
    with open(file_name, "wb") as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    zipfile = ZipFile(file_name)
    zipfile.extractall(path=ppath)
    zipfile.close() 
    progress_bar.close()
    os.remove(file_name)


def check_predictors(ppath, kernel_predictors):
    """
    @params:

    model: a pytorch/onnx/tensorflow model object or a str containing path to the model file
    """
    print("checking local kernel predictors at " + ppath)
    if os.path.isdir(ppath):
        filenames = glob(os.path.join(ppath, "**.pkl"))
        # check if all the pkl files are included
        for kp in kernel_predictors:
            fullpath = os.path.join(ppath, kp + ".pkl")
            if fullpath not in filenames:
                return False
        return True
    else:
        return False
