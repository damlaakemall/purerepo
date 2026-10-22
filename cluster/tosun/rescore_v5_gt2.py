"""Copy frozen v5 answers into a new GT revision; never regenerate or overwrite v5.

Run once per assigned model/shard before submitting score/package stages. GT is
recomputed for every answer, so finished v5 models follow the same deadline too.
"""
import argparse
import json
import shutil
import tempfile
from pathlib import Path

from llm_routing.config import load_benchmark_config
from llm_routing.collection.collect import job_spec, response_hash
from llm_routing.collection.dataset import dataset_dir, load_dataset, specification
from llm_routing.collection.io import digest, environment, file_hash, locked, read_jsonl, seal, write_json


def prepare(old, new, model, member, shard, count):
    before, after = specification(old), specification(new)
    assert {k: v for k, v in before.items() if k not in ('name', 'version')} == {
        k: v for k, v in after.items() if k not in ('name', 'version')}, 'query protocol changed'
    rows, manifest = load_dataset(old)
    target = dataset_dir(new)
    assert target.resolve() != dataset_dir(old).resolve(), 'new version needs a new path'
    target.parent.mkdir(parents=True, exist_ok=True)
    with locked(target.parent / '.dataset.lock'):
        if not target.exists():
            with tempfile.TemporaryDirectory(dir=target.parent) as directory:
                temporary = Path(directory) / 'dataset'
                shutil.copytree(dataset_dir(old), temporary)
                write_json(temporary / 'config.json', after)
                seal(temporary, {k: v for k, v in manifest.items() if k != 'outputs'} |
                     {'spec_hash': digest(after)})
                temporary.rename(target)
    assert load_dataset(new)[0] == rows, 'frozen questions/splits differ'
    source, _, oldspec = job_spec(old, model, member, shard, count)
    destination, selected, newspec = job_spec(new, model, member, shard, count)
    assert oldspec == newspec, 'generation protocol changed'
    if (source / '.writer.lock').exists():
        raise ValueError('source writer lock exists; verify its owning job before recovery')
    original = json.loads((source / 'run.json').read_text())
    assert original['run_id'] == digest(newspec)
    responses = read_jsonl(source / 'responses.jsonl')
    expected = {(q['id'], sample) for q in selected for sample in range(newspec['generation']['k'])}
    assert len(responses) == len(expected), 'source generation must be complete without duplicate attempts'
    assert {(r['query_id'], r['sample_id']) for r in responses} == expected
    assert all(r['success'] and r['response_hash'] == response_hash(r) and
               r['run_id'] == original['run_id'] for r in responses)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with locked(destination.parent / (destination.name + '.import.lock')):
        if destination.exists():
            raise ValueError('destination already exists; resume its scoring, do not import again')
        with tempfile.TemporaryDirectory(dir=destination.parent) as directory:
            temporary = Path(directory) / 'contribution'
            temporary.mkdir()
            shutil.copyfile(source / 'responses.jsonl', temporary / 'responses.jsonl')
            write_json(temporary / 'run.json', {
                'run_id': digest(newspec), 'spec': newspec, 'environment': environment(),
                'generation_origin': {'run': original, 'source_path': str(source),
                                      'responses_sha256': file_hash(source / 'responses.jsonl')}})
            # Same reward protocol: preserve existing records; score() resumes missing ones.
            for channel in ('armorm', 'bartscore'):
                assert old['scoring'].get(channel) == new['scoring'].get(channel)
                artifact = source / (channel + '.jsonl')
                if artifact.exists():
                    shutil.copyfile(artifact, temporary / artifact.name)
            temporary.rename(destination)
    return destination


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model')
    parser.add_argument('member')
    parser.add_argument('--num-shards', type=int, default=4)
    parser.add_argument('--shard-index', type=int, required=True)
    arguments = parser.parse_args()
    print(prepare(load_benchmark_config('configs/benchmarks/shared_v5.yaml'),
                  load_benchmark_config('configs/benchmarks/shared_v5_gt2.yaml'),
                  arguments.model, arguments.member, arguments.shard_index, arguments.num_shards))
