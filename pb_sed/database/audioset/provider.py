
import dataclasses
from pb_sed.data_preparation.provider import DataProvider
from pb_sed.paths import database_jsons_dir


@dataclasses.dataclass
class AudioSetProvider(DataProvider):

    @classmethod
    def finalize_dogmatic_config(cls, config):
        config['json_path'] = str(database_jsons_dir / 'audioset.json')
        config['validate_set'] = 'eval'
        super().finalize_dogmatic_config(config)
        num_events = 527
        config['train_fetcher']['min_label_diversity_in_batch'] = min(
            num_events, config['train_fetcher']['batch_size']
        )
