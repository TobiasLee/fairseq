import os
import argparse
import re
import time
import uuid
import subprocess
import json


def get_parser():
    parser = argparse.ArgumentParser(description='Experiment Status wrapper')
    parser.add_argument('--name', type=str, help='experiment name')
    parser.add_argument('--script', type=str, help='script command or path')
    parser.add_argument('--result_path', type=str, help='result path')

    return parser


def process_result_file(fp):
    with open(fp, 'r') as f:
        line = f.readlines()[-1]
        match = re.match(r'.*BLEU4 = (\d+\.\d+),.*', line)
        assert match
        return match.group(1)


def process_result(result_path):
    fp = os.path.join(result_path, 'avg_10.txt')
    avg_10_result = process_result_file(fp)

    fp = os.path.join(result_path, 'checkpoint_best.txt')
    ckpt_best_result = process_result_file(fp)

    message = '*BLEU*\nAverage Checkpoint: `{}`\nBest Checkpoint: `{}`'.format(
        avg_10_result, ckpt_best_result)

    return message


def main():
    parser = get_parser()
    args = parser.parse_args()

    identifier = str(uuid.uuid1()).lower()

    start_at = time.time()

    script_output = subprocess.Popen(args.script,
                                     stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT)
    script_output.communicate()

    message = process_result(args.result_path)

    end_at = time.time()

    payload = {
        'type': 'completed',
        'name': args.name,
        'start_at': start_at,
        'end_at': end_at,
        'message': message,
    }
    branch = '{}-completed'.format(identifier)
    with open('{}.json'.format(branch), 'w') as f:
        json.dump(payload, f)

    subprocess.run(['./upload_status.sh', branch])


if __name__ == '__main__':
    main()
