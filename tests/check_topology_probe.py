"""Optional storage must preserve diagnostic values and immutable checkpoints."""
import hashlib
import tempfile
import unittest
import numpy as np
from tests.check_attn_probe import fixture
from screening_analysis.attn_probe import probe,save_probe


class InternalProbeChecks(unittest.TestCase):
    def test_optional_storage_all_window_types(self):
        import jax
        for mode in ('cos','smooth7','band','sin','gauss','fixed','nomask'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as tmp:
                net,p,data,settings,L,ck=fixture(tmp,mode)
                before=hashlib.sha256(ck.read_bytes()).hexdigest()
                base=probe(tmp,settings=settings,lattice=L,nconf=1)
                full=probe(tmp,settings=settings,lattice=L,nconf=1,collect_internal=True)
                for key in ('S','W','Alpha','W_window','d'):
                    np.testing.assert_array_equal(base[key],full[key])
                self.assertEqual(hashlib.sha256(ck.read_bytes()).hexdigest(),before)
                similarity=np.einsum('blnhd,blmhd->blhnm',full['q_normalized'],full['k_normalized'])
                np.testing.assert_allclose(similarity,full['S'],atol=2e-15)
                c=full['head_vectors']
                np.testing.assert_allclose(c.sum(axis=-2)+full['head_bias'][:,:,None,:],full['attention_output'],atol=2e-15)
                np.testing.assert_array_equal(full['hidden_output'][:,0],full['hidden_input'][:,1])
                if mode=='fixed':
                    self.assertNotIn('v',full); self.assertIn('pair_features',full)
                else:self.assertIn('v',full)
                save_probe(full,ck.parent/'internal.npz')
                with np.load(ck.parent/'internal.npz',allow_pickle=False) as z:
                    for key in z.files: z[key]
                jax.clear_caches()


if __name__=='__main__':unittest.main()
