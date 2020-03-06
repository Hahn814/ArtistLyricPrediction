from base.environment import env
from utils.text import Corpus


class Track(Corpus):

    def __init__(self, title, artist, lyrics=None, album=None, year=None):
        super().__init__(corpus=lyrics)
        self.logger = env.get_logger(self.__class__.__name__)
        self.__title = title
        self.__artist = artist
        self.__album = album
        self.__year = year
        self.logger.debug("New artist instantiated: {}".format(self))

    def __str__(self):
        return "{classname} - Artist: {artist}, Title: {title}, Album: {album}, Year: {year}".format(classname=self.__class__.__name__, artist=self.__artist, title=self.__title, album=self.__album, year=self.__year)

    @property
    def year(self):
        return self.__year

    @year.setter
    def year(self, year):
        self.__year = year

    @property
    def album(self):
        return self.__album

    @album.setter
    def album(self, album):
        self.__album = album

    @property
    def lyrics(self):
        return self.corpus

    @lyrics.setter
    def lyrics(self, lyrics):
        self.corpus = lyrics
        self.logger.debug("Updated Track lyrics: {}".format(self.lyrics))

    @property
    def title(self):
        return self.__title

    @title.setter
    def title(self, title):
        og = self.__title
        self.__title = title
        self.logger.debug("Updated title: '{}' -> '{}'".format(og, self.title))

    @property
    def artist(self):
        return self.__artist

    @artist.setter
    def artist(self, artist):
        og = self.artist
        self.__artist = artist
        self.logger.debug("Updated artist: '{}' -> '{}'".format(og, self.artist))

    def __hash__(self):
        return self.artist, self.title

    def keys(self):
        return ['Artist', 'Album', 'Song', 'Year', 'Lyric Count', 'Sentiment Score']

    def __getitem__(self, item):
        val = None
        if item in self.keys():
            if item == 'Artist':
                val = self.artist
            elif item == 'Album':
                val = self.album
            elif item == 'Song':
                val = self.title
            elif item == 'Year':
                val = self.year
            elif item == 'Lyric Count':
                val = len(self.corpus.split())
            elif item == 'Sentiment Score':
                val = self.get_average_sentiment_score()
        else:
            raise KeyError("Invalid key provided: {}".format(item))

        return val
