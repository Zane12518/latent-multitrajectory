"""Lightweight public-document, evidence and accidental-sensitive-text checks."""
import argparse
import csv
import json
from pathlib import Path
import re
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def check():
    patterns = [r'/(?:Users|inspire)/', r'https?://[^\s]*feishu\.cn/',
                r'gh[pousr]_[A-Za-z0-9]{25,}', r'github_pat_[A-Za-z0-9_]{30,}',
                r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----']
    files = [p for folder in ('docs','src','training','evaluation','scripts','tests','configs','.github','legacy')
             for p in (ROOT/folder).rglob('*') if p.is_file() and p.suffix in ('.md','.py','.json','.csv','.yml')]
    files += list(ROOT.glob('*.md'))
    for p in files:
        text=p.read_text()
        if any(re.search(pattern,text) for pattern in patterns):
            raise ValueError(f'sensitive-text candidate: {p.relative_to(ROOT)}')
        if p.suffix=='.md':
            for link in re.findall(r'\]\(([^\s)]+)\)',text):
                if '://' in link or link.startswith('#'):continue
                target=(p.parent/link.split('#')[0]).resolve()
                if not target.exists():raise ValueError(f'broken relative link: {p.relative_to(ROOT)} -> {link}')
    for task in ('sat','maze'):
        d=ROOT/'docs/evidence/gru'
        s=json.loads((d/f'{task}_summary.json').read_text())
        assert len(s['primary_set_minus_multi'])==6
        for pair,effect in s['primary_set_minus_multi'].items():
            a,b=pair.split(' minus ')
            for metric in ('U','validity','first','selected'):
                assert abs(s['models'][a][metric]-s['models'][b][metric]-effect[metric]['delta'])<1e-10
            assert effect['U']['sim95'][0]>0
            assert all(effect[k]['sim95'][0]>-.01 for k in ('validity','first','selected'))
        rows=list(csv.DictReader((d/f'{task}_all_models_K_curves.csv').open()))
        assert {r['model'] for r in rows}==set(s['models'])
    return dict(status='PASS',text_files=len(files),gru_pairs=12,
                scope='relative links, explicit sensitive patterns and aggregate evidence consistency; not a comprehensive secret audit or raw-prediction recomputation')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--links',action='store_true');a=p.parse_args()
    result=check()
    if a.links:
        urls=['https://github.com/Zane12518/latent-multitrajectory',
              'https://github.com/Zane12518/latent-multitrajectory/releases/tag/v0.1.0',
              'https://zane12518.github.io/latent-multitrajectory/demo/']
        for url in urls:
            with urllib.request.urlopen(url,timeout=30) as response:
                assert response.status==200
        result['public_links']=len(urls)
    print(json.dumps(result,indent=2))
