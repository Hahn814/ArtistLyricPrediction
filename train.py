#!interpreter [optional-arg]
# -*- coding: utf-8 -*-

"""
{Description}
{License_info}
"""

# Futures
from __future__ import print_function
# […]

# Built-in/Generic Imports
import os
import sys
import argparse
import logging
import lyricsgenius

# […]

# […]

# Own modules
from GetLyrics.tk.track import Track
from GetLyrics.utils import text

# […]

__author__ = 'Paul Hahn'


logger = logging.getLogger(__name__)


def get_arg_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    filter_parser = parser.add_argument_group('Filter arguments')
    filter_parser.add_argument('--artist', default=None, action='append', help='The name of the artist')
    filter_parser.add_argument('--max_tracks', default=100, type=int, help='Upper limit to how many tracks each artist can have')

    return parser


def get_args():
    arg_parser = get_arg_parser()
    args_parsed = arg_parser.parse_args()

    return args_parsed


if __name__ == '__main__':
    args = get_args()
    artist_names = args.artist
    max_results = args.max_tracks

    genius = lyricsgenius.Genius("nNhT-omaJrPgrGxd4w_qHHWXXUcTdqZMnQ7v_gr5lohgHYlYlm623WgzdjDRikNv")
    if artist_names:
        for artist_name in artist_names:
            artist = genius.search_artist(artist_name, max_songs=max_results)

            for song in artist.songs:
                if song.lyrics:
                    logger.info("Found: {}, Limit: {}".format(len(artist.songs), max_results))
                    track = Track(title=song.title, lyrics=song.lyrics, artist=song.artist, album=song.album, year=song.year)
                    lyrics = track.lyrics
                    logger.debug("'{}': '{}'".format(track.title, lyrics))
