# -*- coding: utf-8 -*-
"""
Module: environment.py
Created by: Paul Hahn
Created date: 2019-03-31

Standards shared by multiple modules throughout the package.
"""

import logging
import sys
import csv

FORMAT = "%(asctime)s - %(name)s - %(levelname)s: %(message)s"
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=FORMAT)


class env(object):

    def __init__(self):
        self.logger = logging.getLogger(name=self.__class__.__name__)

    @staticmethod
    def get_logger(context, level=logging.DEBUG):
        logger = logging.getLogger(name=context)
        logger.setLevel(level=level)
        return logger

    @staticmethod
    def get_csv_writer(file_path, fieldnames, write_header=True):
        f = open(file_path, 'w')
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator='\n')
        if write_header:
            writer.writeheader()

        return writer
