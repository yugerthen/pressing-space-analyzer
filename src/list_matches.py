from SoccerNet.Downloader import getListGames

games = getListGames(split="train")
laliga_games = [g for g in games if "spain_laliga" in g]

print(f"{len(laliga_games)} matchs de Liga trouvés\n")
for g in laliga_games[:15]:
    print(g)