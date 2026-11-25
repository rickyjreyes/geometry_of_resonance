#!/usr/bin/env python3
"""Explicit full-local and public-CI scopes. Never overwrite an execution receipt."""
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True)
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--public-ci',action='store_true');mode.add_argument('--d4-source-root',type=Path)
    p.add_argument('--source-extraction-receipt',type=Path);args=p.parse_args()
    if args.public_ci and args.source_extraction_receipt:p.error('public CI does not extract private sources')
    out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'execution.json').exists():raise ValueError('Choose a fresh output directory')
    py=sys.executable;cross=out/'cross_paper';cross.mkdir()
    shutil.copyfile(ROOT/'audit/cross_paper/verify_derivations.py',cross/'verify.py')
    commands=[
      ('phase-v-identities',[py,'derivations/d4_binding.py','--output',str(out/'verification.json')]),
      ('phase-v-adversarial',[py,'audit/d4_binding/run_diagnostics.py','--output',str(out/'diagnostics.json')]),
      ('all-derivation-regressions',[py,'-m','unittest','discover','-v','-s','derivations','-p','test_*.py']),
      ('pr35-identities',[py,'derivations/strict_binding.py','--output',str(out/'pr35_verification.json')]),
      ('pr35-adversarial',[py,'audit/strict_binding/run_diagnostics.py','--output',str(out/'pr35_diagnostics.json')]),
      ('pr34-identities',[py,'derivations/foundational_recovery.py','--output',str(out/'pr34_verification.json')]),
      ('pr33-identities',[py,'derivations/foundational_closure.py','--output',str(out/'pr33_verification.json')]),
      ('cross-paper-replay',[py,str(cross/'verify.py')]),
      ('pr33-integrity',[py,'audit/foundational/validate_records.py']),
      ('pr34-integrity',[py,'audit/foundational_recovery/validate_records.py']),
      ('pr35-integrity',[py,'audit/strict_binding/validate_records.py']),
      ('historical-d4-scope',[py,'audit/d4_handoff/validate_claims.py']),
      ('prospective-gate-self-test',[py,'audit/prospective/freeze_gate.py','--self-test']),
      ('prospective-regressions',[py,'-m','unittest','discover','-v','-s','audit/prospective','-p','test_*.py'])]
    if args.public_ci:commands.append(('phase-v-recorded-evidence',[py,'audit/d4_binding/validate_records.py']))
    else:
        private=out/'d4_source'
        commands.extend([
          ('prepare-pinned-d4-inputs',[py,'audit/strict_binding/prepare_d4_replay.py','--destination',str(private),'--from-root',str(args.d4_source_root.resolve())]),
          ('fresh-source-conventions',[py,'audit/d4_binding/verify_sources.py','--source-root',str(args.d4_source_root.resolve()),'--output',str(out/'source_verification.json')]),
          ('d4-rational-replay',[py,str(private/'tools/closure_d4_basin_rational_replay.py'),'--output',str(out/'d4_basin_replay.json')])])
    receipt={'schema':'d4-binding-execution-v1','frozen_base':'65499cabca7e177286b6541fa1a5a00d51a50ba5',
      'date_local':'2026-10-23 America/Los_Angeles','mode':'public-ci' if args.public_ci else 'full-local',
      'python':platform.python_version(),'dependencies':{n:importlib.metadata.version(n) for n in ['sympy','mpmath','numpy','scipy']},
      'implementation_sha256':{str(q.relative_to(ROOT)):hashlib.sha256(q.read_bytes()).hexdigest() for q in [HERE/'reproduce.py',HERE/'run_diagnostics.py',HERE/'verify_sources.py',HERE/'source_ledger.json',ROOT/'derivations/d4_binding.py',ROOT/'derivations/test_d4_binding.py']},
      'commands':[],'all_commands_passed':False,
      'scope':{'D4_private_inputs_hash_checked':False,'D4_rational_replay_executed':False,'private_source_conventions_checked':False,'recorded_evidence_validated':False,'fresh_authenticated_source_extraction_receipt_included':False,'original_D4_interval_campaign_rerun':False,'PDE_propagation':False,'stationary_root_search':False,'external_mathematical_review':False,'physical_spectrum':False}}
    if args.source_extraction_receipt:
        origin=json.loads(args.source_extraction_receipt.read_text());ledger=json.loads((HERE/'source_ledger.json').read_text())
        assert origin['revision']=='051b6eca10c46b8e038993d60cabb20715194a20'
        assert {r['path']:r['sha256'] for r in origin['files']}=={r['path']:r['sha256'] for r in ledger['d4_replay_inputs']}
        shutil.copyfile(args.source_extraction_receipt,out/'source_extraction.json')
        receipt['source_extraction_sha256']=hashlib.sha256((out/'source_extraction.json').read_bytes()).hexdigest()
        receipt['scope']['fresh_authenticated_source_extraction_receipt_included']=True
    scopes={'prepare-pinned-d4-inputs':'D4_private_inputs_hash_checked','fresh-source-conventions':'private_source_conventions_checked','d4-rational-replay':'D4_rational_replay_executed','phase-v-recorded-evidence':'recorded_evidence_validated'}
    for name,command in commands:
        started=datetime.now(timezone.utc).isoformat();tick=time.monotonic()
        done=subprocess.run(command,cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        log=out/(name+'.log');log.write_text(done.stdout)
        receipt['commands'].append({'name':name,'command':command,'cwd':str(ROOT),'started_at_utc':started,'seconds':round(time.monotonic()-tick,3),'returncode':done.returncode,'log':log.name,'log_sha256':hashlib.sha256(log.read_bytes()).hexdigest()})
        if done.returncode==0 and name in scopes:receipt['scope'][scopes[name]]=True
        (out/'execution.json').write_text(json.dumps(receipt,indent=2)+'\n')
        print(name+(': PASS' if done.returncode==0 else ': FAIL'),flush=True)
        if done.returncode:print(done.stdout);raise SystemExit(done.returncode)
    receipt['all_commands_passed']=True;(out/'execution.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(f'All {len(commands)} command groups passed.')
    if args.public_ci:print('Public CI validates recorded private-source D4 evidence; it does NOT execute the private-source replay.')
    else:print('All five D4 rational gates freshly executed; original interval root/coercivity campaigns were not rerun.')


if __name__=='__main__':main()
