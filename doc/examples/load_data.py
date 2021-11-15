from precovery import precovery_db

db = precovery_db.PrecoveryDatabase.from_dir("db", create=True)
db.frames.load_hdf5("data_file.hdf5")
