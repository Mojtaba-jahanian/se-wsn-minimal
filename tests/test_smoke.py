import yaml
from sewsn.evaluation.run_experiment import run_experiment

def test_smoke_run():
    with open('configs/quick.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    cfg['experiment']['rounds'] = 5
    logs, summary = run_experiment(cfg, seeds=[0])
    assert not logs.empty
    assert set(summary['method']) == set(cfg['experiment']['methods'])
