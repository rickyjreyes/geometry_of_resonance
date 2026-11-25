import copy,hashlib,json,unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from freeze_gate import check,verify_artifacts,TEMPLATE

class FreezeGateTests(unittest.TestCase):
    def test_nonfinite_reversed_and_boolean_targets(self):
        for lo,hi in [(float('nan'),1),(0,float('inf')),(2,1),(False,True),('1','2')]:
            spec=json.loads(TEMPLATE.read_text());spec['theory']['dimensionless_target']={'lower':lo,'upper':hi}
            self.assertTrue(any('target' in b for b in check(spec)['blockers']))
    def test_malformed_shapes_fail_closed(self):
        for spec in [None,[],{'theory':[1],'protocol':[2],'attestation':3,'execution':'x'}]:
            self.assertEqual(check(spec)['status'],'BLOCKED')
    def test_byte_mutation_and_manifest_disagreement(self):
        with TemporaryDirectory() as d:
            p=Path(d)/'manifest.txt';p.write_text('opaque IDs only\n');sha=hashlib.sha256(p.read_bytes()).hexdigest()
            spec={'protocol':{'source_manifest_sha256':sha},'artifacts':[{'role':r,'path':p.name,'sha256':sha} for r in ['source_manifest','derivation','detector_forward_map','protocol']]}
            self.assertFalse(verify_artifacts(spec,d)[0])
            p.write_text('changed\n');self.assertTrue(any('SHA256 mismatch' in b for b in verify_artifacts(spec,d)[0]))
            p.write_text('opaque IDs only\n');spec['protocol']['source_manifest_sha256']='0'*64
            self.assertIn('source manifest digest disagrees with protocol',verify_artifacts(spec,d)[0])
    def test_outcome_roles_and_path_escape_are_rejected(self):
        with TemporaryDirectory() as d:
            spec={'artifacts':[{'role':'holdout_outcomes','path':'secret','sha256':'0'*64},{'role':'protocol','path':'../secret','sha256':'0'*64}]}
            blockers,receipts=verify_artifacts(spec,d)
            self.assertEqual(receipts,[])
            self.assertTrue(any('escapes' in b for b in blockers))
            self.assertTrue(any('outcomes' in b for b in blockers))
    def test_template_never_executes_holdout(self):
        out=check(json.loads(TEMPLATE.read_text()))
        self.assertEqual(out['status'],'BLOCKED')
        self.assertTrue(out['no_holdout_executed_by_checker'])
        self.assertTrue(out['no_claim_of_prospective_prediction'])
if __name__=='__main__':unittest.main()
