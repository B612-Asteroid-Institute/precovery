from precovery import precovery_db

db = precovery_db.PrecoveryDatabase.from_dir("db", create=True)
db.frames.load_csv("data_file.csv")
