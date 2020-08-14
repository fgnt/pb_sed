
import soundfile
import concurrent.futures
import itertools


def prepare_sound_dataset(examples, max_examples=int(1e12), postprocess_fn=None):
    dataset = {}
    with concurrent.futures.ThreadPoolExecutor() as ex:
        for _, example_id, example in itertools.islice(
                filter(
                    lambda x: x[0], ex.map(prepare_sound_example, examples.items())
                ),
                max_examples
        ):
            if postprocess_fn is not None:
                example = postprocess_fn(example)
            dataset[example_id] = example
    return dataset


def prepare_sound_example(item: (str, dict)) -> (bool, str, dict):
    """
    Creates example dict for one example with example_id.
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
        return False, example_id, None
