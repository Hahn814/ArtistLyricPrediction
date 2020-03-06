import os
import re
import argparse
import lyricsgenius
from collections import Counter
from base.environment import env
from tk.track import Track

logger = env.get_logger(__name__)
HEADER = ['Artist', 'Album', 'Song', 'Year', 'Lyric Count', 'Sentiment Score', 'Word', 'POS']
IS_WORD = re.compile(r'\w+')


def get_arg_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    filter_parser = parser.add_argument_group('Filter the results')
    filter_parser.add_argument('--artist', default=None, action='append', help='The name of the artist')

    out_parser = parser.add_argument_group('Output Parameters')
    out_parser.add_argument('--max_results', default=500, help='Limit the number of results to consider')
    out_parser.add_argument('--out', default='genius.csv', help='CSV results file')

    return parser


def get_args():
    arg_parser = get_arg_parser()
    args_parsed = arg_parser.parse_args()

    return args_parsed

def is_word(word):
    return IS_WORD.match(word)


if __name__ == '__main__':
    args = get_args()
    artist_names = args.artist
    max_results = int(args.max_results)
    outfile = env.get_csv_writer(os.path.abspath(args.out), fieldnames=HEADER)
    genius = lyricsgenius.Genius("nNhT-omaJrPgrGxd4w_qHHWXXUcTdqZMnQ7v_gr5lohgHYlYlm623WgzdjDRikNv")

    if artist_names:
        for artist_name in artist_names:
            artist = genius.search_artist(artist_name, max_songs=max_results)
            logger.info("Found: {}, Limit: {}".format(len(artist.songs), max_results))

            for song in artist.songs:
                if song.lyrics:
                    track = Track(title=song.title, lyrics=song.lyrics, artist=song.artist, album=song.album, year=song.year)
                    avg_sentiment_score = track.get_average_sentiment_score()
                    lyric_count = len(track)
                    lyric_counter = Counter(track.lyrics.split())
                    c_words = track.get_most_common()
                    c_tags = track.get_most_common(tags=True)
                    tags = c_tags.most_common()
                    for (word, tag), c in tags:
                        res = dict(track)
                        res.update({'Word': word, 'POS': tag})
                        outfile.writerow(rowdict=res)
                else:
                    logger.warning("Song has no lyrics: '{}': '{}' - '{}'".format(song.artist, song.album, song.title))
