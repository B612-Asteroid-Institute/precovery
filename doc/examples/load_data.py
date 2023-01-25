from precovery import precovery_db

db = precovery_db.PrecoveryDatabase.from_dir("db", create=True)
db.frame.add_dataset("example-dataset")
db.frames.load_csv("data_file.csv", "example-dataset")
