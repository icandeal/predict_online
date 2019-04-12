# -*- coding: utf-8 -*-

import os
import sys
import oss2
import time
import zipfile
import importlib

# oss Endpoint
endpoint = os.environ['Endpoint']
# oss Applibs bucket
app_lib_bucket = os.environ['AppLibsBucket']
# lib包本地存放位置
app_lib_local_dir = os.environ['AppLibLocalDir']
# lib包oss存放位置
app_lib_oss_dir = os.environ['AppLibOSSDir']

# oss Models bucket
model_bucket = os.environ['ModelsBucket']
# model本地存放位置
model_object = os.environ['ModelOSSDir']
# model oss存放位置
model_dir = os.environ['ModelLocalDir']

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
local = bool(os.getenv('local', ""))

print('local running: ' + str(local))

initialized = False

HELLO_WORLD = b"Untouched!!\n"


def download_and_unzip_if_not_exist(bucket, objectKey, path, context):
    creds = context.credentials

    if local:
        print('thank you for running function in local!!!!!!')
        auth = oss2.Auth(creds.access_key_id,
                         creds.access_key_secret)
    else:
        auth = oss2.StsAuth(creds.access_key_id,
                            creds.access_key_secret,
                            creds.security_token)

    print('objectKey: ' + objectKey)
    print('path: ' + path)
    print('endpoint: ' + endpoint)
    print('bucket: ' + bucket)

    bucketObj = oss2.Bucket(auth, endpoint, bucket)

    zipName = '/tmp/tmp.zip'

    print('before downloading ' + objectKey + ' ...')
    start_download_time = time.time()
    bucketObj.get_object_to_file(objectKey, zipName)
    print('after downloading, used %s seconds...' % (time.time() - start_download_time))

    if not os.path.exists(path):
        os.mkdir(path)

    print('before unzipping ' + objectKey + ' ...')
    start_unzip_time = time.time()
    with zipfile.ZipFile(zipName, "r") as z:
        z.extractall(path)
    print('unzipping done, used %s seconds...' % (time.time() - start_unzip_time))


def initializer(context):
    download_and_unzip_if_not_exist(app_lib_bucket, app_lib_oss_dir, app_lib_local_dir, context)
    download_and_unzip_if_not_exist(model_bucket, model_object, model_dir, context)
    if not local:
        sys.path.insert(1, app_lib_local_dir)
    print(sys.path)


def handler(environ, start_response):
    global HELLO_WORLD
    global initialized

    index_module = importlib.import_module(".", "index")
    index_handler = getattr(index_module, "do_predict")
    initialized, result = index_handler(environ, initialized)

    if result:
        HELLO_WORLD = b"Success!!\n"
    else:
        HELLO_WORLD = b"Failed!!\n"

    status = '200 OK'
    response_headers = [('Content-type', 'text/plain')]
    start_response(status, response_headers)
    return [HELLO_WORLD]
