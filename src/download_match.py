from SoccerNet.Downloader import SoccerNetDownloader
import os

downloader = SoccerNetDownloader(LocalDirectory="../data/soccernet")
downloader.password = os.environ["SOCCERNET_PASSWORD"]

downloader.downloadGame(
    game="spain_laliga/2014-2015/2015-02-21 - 18-00 Barcelona 0 - 1 Malaga",
    files=["1_224p.mkv"]
)