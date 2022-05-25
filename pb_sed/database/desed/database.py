
from lazy_dataset.database import JsonDatabase
from pb_sed.paths import database_jsons_dir


class DESED(JsonDatabase):
    def __init__(self, json_path=database_jsons_dir / 'desed.json'):
        super().__init__(json_path=json_path)
