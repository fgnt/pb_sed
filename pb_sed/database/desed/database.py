
from lazy_dataset.database import Database
from paderbox.io.json_module import load_json
from cached_property import cached_property
from pb_sed.paths import database_jsons_dir


class DESED(Database):
    def __init__(self, json_path=database_jsons_dir / 'desed.json'):
        self._json_path = json_path
        super().__init__()

    @cached_property
    def data(self):
        return load_json(self._json_path)
