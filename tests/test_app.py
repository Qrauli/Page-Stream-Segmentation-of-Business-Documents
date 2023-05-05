from unittest import mock

from fastapi import status
from fastapi.testclient import TestClient

import app as app_module
from app import app

from io import BytesIO
from zipfile import ZipFile


def test_call_predict_endpoint_one_page():
    test_response = [1]
    # The line below will "replace" the result of `call_api()` with whatever
    # is given in `return_value`. The original function is never executed.
    #with mock.patch.object(app_module, "predict", return_value=test_response) as mock_predict:
    with TestClient(app) as client:
        test_file = 'Doc1.pdf'
        # iles = {'file': ('Doc1.pdf', open(test_file, 'rb'))}
        res = client.post("/predict", files={"file": open(test_file, 'rb')})

    assert res.status_code == status.HTTP_200_OK
    assert res.headers["Content-Type"] == "application/zip"
    # assert res.json() == "some_value"
    zip_file = BytesIO(res.content)

    # Check that the BytesIO instance can be used to create a valid ZIP file
    with ZipFile(zip_file, "r") as z:
        assert len(z.filelist) == 1

def test_call_predict_endpoint_multiple_pages():
    test_response = [1, 1, 1]
    # The line below will "replace" the result of `call_api()` with whatever
    # is given in `return_value`. The original function is never executed.
    #with mock.patch.object(app_module, "predict", return_value=test_response) as mock_predict:
    with TestClient(app) as client:
        test_file = 'Doc2.pdf'
        # iles = {'file': ('Doc1.pdf', open(test_file, 'rb'))}
        res = client.post("/predict", files={"file": open(test_file, 'rb')})

    assert res.status_code == status.HTTP_200_OK
    assert res.headers["Content-Type"] == "application/zip"
    # assert res.json() == "some_value"
    zip_file = BytesIO(res.content)

    # Check that the BytesIO instance can be used to create a valid ZIP file
    with ZipFile(zip_file, "r") as z:
        assert len(z.filelist) >= 1

def test_call_predict_endpoint_wrong_format():
    test_response = [1]
    # The line below will "replace" the result of `call_api()` with whatever
    # is given in `return_value`. The original function is never executed.
    #with mock.patch.object(app_module, "predict", return_value=test_response) as mock_predict:
    with TestClient(app) as client:
        test_file = 'Doc1.txt'
        # iles = {'file': ('Doc1.pdf', open(test_file, 'rb'))}
        res = client.post("/predict", files={"file": open(test_file, 'rb')})

    assert res.status_code == status.HTTP_400_BAD_REQUEST