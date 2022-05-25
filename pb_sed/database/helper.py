
import soundfile
import concurrent.futures
from tqdm import tqdm


def prepare_sound_dataset(examples, postprocess_fn=None):
    """
    filters unavailable audio files and adds audio length to examples
    Args:
        examples:
        max_examples:
        postprocess_fn:

    Returns:

    """
    dataset = {}
    missing = set()
    with concurrent.futures.ThreadPoolExecutor() as ex:
        for available, example_id, example in tqdm(
                ex.map(prepare_sound_example, examples.items()),
                total=len(examples)
        ):
            if not available:
                missing.add(example_id)
            if postprocess_fn is not None:
                example = postprocess_fn(example)
            dataset[example_id] = example
    return dataset, missing


def prepare_sound_example(item: (str, dict)) -> (bool, str, dict):
    """
    Adds audio length to example dict.
    """
    example_id, example = item
    audio_path = example['audio_path']
    try:
        with soundfile.SoundFile(str(audio_path)) as f:
            length = len(f) / f.samplerate
    except:
        length = 0.
    if length > 0.:
        example['audio_length'] = length
        return True, example_id, example
    else:
        example.pop('audio_path')
        return False, example_id, example
