"""正常课程平地训练后自动验收；不会自动启动坡面长训练。"""
import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-dir', required=True)
    args = parser.parse_args()
    scripts = Path(__file__).resolve().parent
    output = Path(args.save_dir).resolve()
    if output.exists():
        raise FileExistsError(f'请为本次试验使用新目录: {output}')
    output.mkdir(parents=True)
    with (output / 'training.log').open('w', encoding='utf-8') as log:
        subprocess.run([sys.executable, '-u', str(scripts / 'train_ppo.py'),
                        '--iterations', '1200', '--bounded-policy', '--target-kl', '0.01',
                        '--device', 'cuda', '--save-every', '250', '--save-dir', str(output)],
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    summaries = []
    for label, command in [('forward', .15), ('backward', -.15), ('standing', 0.0)]:
        result = output / ('eval_' + label)
        subprocess.run([sys.executable, str(scripts / 'validate_stage4.py'),
                        '--checkpoint', str(output / 'ppo_final.pt'), '--episodes', '100',
                        '--theta', '0', '--prob-attach', '1', '--iteration', '0',
                        '--command', str(command), '--output', str(result)], check=True)
        summary = json.loads(result.with_suffix('.json').read_text(encoding='utf-8'))
        # Engineering screening thresholds, not paper acceptance criteria.
        passed = summary['survival_rate'] >= .95 and summary['mean_velocity_rmse'] < .1
        if command:
            passed &= summary['mean_displacement_m'] * (1 if command > 0 else -1) >= 1.0
        else:
            passed &= abs(summary['mean_displacement_m']) < .1
        summaries.append(dict(label=label, passed=bool(passed), metrics=summary))
    report = dict(status='evaluated', passed=all(s['passed'] for s in summaries),
                  note='工程筛查：存活>=95%、RMSE<0.1；正反向10秒位移>=1米；静止位移<0.1米。不是论文复现验收。确定性平地100回合初态相同，不能作为独立随机鲁棒性样本。',
                  evaluations=summaries)
    (output / 'assessment.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(status=report['status'], passed=report['passed'])), flush=True)


if __name__ == '__main__':
    main()
