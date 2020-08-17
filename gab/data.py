#!/usr/bin/env python3
# -*- coding: utf-8 -*

def download_data():

    https://cocodataset.org/#download
    1. download img & annotation uisng gsutil
        mkdir val2017
        gsutil -m rsync gs://images.cocodataset.org/val2017 val2017

    2. use COCO api
    