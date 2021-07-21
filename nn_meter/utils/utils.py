# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
from zipfile import ZipFile
from tqdm import tqdm
import requests
from packaging import version


def download_from_url(urladdr, ppath):
    """
    download the kernel predictors from the url
    @params:

    urladdr: github release url address
    ppath: the targeting hardware_inferenceframework name

    """
    file_name = ppath + "/" + ".zip"
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
    progress_bar.close()
    os.remove(file_name)


def try_import_onnx(require_version = "1.9.0"):
    try:
        import onnx
        if version.parse(onnx.__version__) != version.parse(require_version):
            print(f'WARNING: onnx=={onnx.__version__} is not well tested now, well tested version: onnx=={require_version}' )
        return onnx
    except ImportError:
        print(f'You have not install the onnx package, please install onnx=={require_version} and try again.')
        exit()


def try_import_torch(require_version = "1.8.1"):
    try:
        import torch
        if version.parse(torch.__version__) != version.parse(require_version):
            print(f'WARNING: torch=={torch.__version__} is not well tested now, well tested version: torch=={require_version}' )
        return torch
    except ImportError:
        print(f'You have not install the torch package, please install torch=={require_version} and try again.')
        exit()


def try_import_tensorflow(require_version = "1.9.0"):
    try:
        import tensorflow
        if version.parse(tensorflow.__version__) != version.parse(require_version):
            print(f'WARNING: tensorflow=={tensorflow.__version__} is not well tested now, well tested version: tensorflow=={require_version}' )
        return tensorflow
    except ImportError:
        print(f'You have not install the tensorflow package, please install tensorflow=={require_version} and try again.')
        exit()


def try_import_torchvision_models():
    try:
        import torchvision
        return torchvision.models
    except ImportError:
        print(f'You have not install the torchvision package, please install torchvision and try again.')
        exit()
        