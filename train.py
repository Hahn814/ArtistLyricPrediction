# -*- coding: utf-8 -*-
__author__ = 'Paul Hahn'

import argparse
import logging
import sys

import pandas as pd
from Model.Model import TrainingBundle, PredictiveModel

FORMAT = "%(asctime)s - %(name)s - %(levelname)s: %(message)s"
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=FORMAT)


"""
    The header of the incoming data is [artist, song, link, text]
    Collected from https://www.kaggle.com/mousehead/songlyrics/data
"""
ARTIST_COL_NAME = 'artist'
SONG_COL_NAME = 'song'
LINK_COL_NAME = 'link'
TEXT_COL_NAME = 'text'


logger = logging.getLogger(__name__)


def get_arg_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input', default=None, action='append', help='Input csv to be processed')
    parser.add_argument('--output', default='model.json', help='Where to persist the model JSON data')
    parser.add_argument('--artist_name', default=None, action='append', help='Filter which artists you want to train on')

    return parser


def get_args():
    arg_parser = get_arg_parser()
    args_parsed = arg_parser.parse_args()

    return args_parsed


if __name__ == '__main__':
    args = get_args()
    input_files = args.input
    output_file = args.output
    artist_names = args.artist_name
    models = {}

    """Read the data into a DataFrame"""
    for _inp_fp in input_files:
        df = pd.read_csv(_inp_fp, delimiter=',')
        for a in artist_names:
            artist_data: pd.DataFrame = df.loc[df['artist'] == a]
            logger.info("Processing artist: '{}'".format(a))

            if not artist_data.empty:
                _tb = TrainingBundle(data=artist_data[TEXT_COL_NAME].tolist(), meta={'artist': a})
                logger.info("TrainingBundle created: {}".format(_tb))
                logger.debug("Training the model...")
                model = PredictiveModel(training_bundle=_tb)
                logger.debug("Model training complete")
                logger.debug("Writing model to file: '{}'".format(output_file))
                model.write_model_to_file(output_file)
            else:
                logger.warning("Artist has no data: '{}'".format(a))


