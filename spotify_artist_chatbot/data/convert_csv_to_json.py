# Source - https://stackoverflow.com/q
# Posted by BeanBagKing, modified by community. See post 'Timeline' for change history
# Retrieved 2026-01-19, License - CC BY-SA 3.0

import csv
import json

csvfile = open("test_artists.csv", "r")
jsonfile = open("../test_artists.json", "w")

fieldnames = ("id", "name", "followers", "popularity", "genres", "main_genre")
reader = csv.DictReader(csvfile, fieldnames)
out = json.dumps([row for row in reader])
jsonfile.write(out)
