import os
import argparse
import re
import slacker


def get_parser():
    parser = argparse.ArgumentParser(description='Slack wrapper for fairseq train/valid')
    parser.add_argument('--name', type=str, help='experiment name')
    parser.add_argument('--script', type=str, help='script path')
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

    token = os.environ['SLACK_TOKEN']
    channel = os.environ['SLACK_CHANNEL']
    manager = slacker.StatusManager(args.name, token, channel)
    manager.on_completed = lambda stdout, stderr: process_result(args.result_path)

    manager.run_script(args.script)


if __name__ == '__main__':
    main()
