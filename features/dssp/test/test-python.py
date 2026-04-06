from mkdssp import dssp, TestBond, helix_type, chain_break_type, helix_type, helix_position_type, structure_type
import gzip
import unittest

class TestDssp(unittest.TestCase):
    def setUp(self):
        with gzip.open("1cbs.cif.gz", "rt") as f:
            file_content = f.read()

        self.dssp = dssp(file_content)
    
    def test_dssp(self):
        stats = self.dssp.statistics
        self.assertEqual(stats.residues, 137)
        
    def test_count(self):
        count = 0
        for r in self.dssp:
            count += 1
        self.assertEqual(count, 137)

    def test_bond(self):
        a = self.dssp.get('A', 137)
        b = self.dssp.get('A', 6)
        self.assertTrue(TestBond(a, b))
    
    def test_one_residue(self):
        r = self.dssp.get("A", 15)

        self.assertEqual(r.asym_id, "A")
        self.assertEqual(r.seq_id, 15)
        
        self.assertEqual(r.compound_letter, "F")
        self.assertEqual(r.auth_asym_id, "A")
        self.assertEqual(r.auth_seq_id, 15)
        self.assertEqual(r.pdb_strand_id, "A")
        self.assertEqual(r.pdb_seq_num, 15)
        self.assertEqual(r.pdb_ins_code, '')
        self.assertAlmostEqual(r.alpha, 44.76425552368164)
        self.assertAlmostEqual(r.kappa, 73.85977935791016)
        self.assertAlmostEqual(r.phi, -66.86814880371094)
        self.assertAlmostEqual(r.psi, -48.56082534790039)
        self.assertAlmostEqual(r.tco, 0.9395495057106018)
        self.assertAlmostEqual(r.omega, 179.16152954101562)
        self.assertEqual(r.is_pre_pro, False)
        self.assertEqual(r.is_cis, False)
        self.assertEqual(r.chiral_volume, 0.0)
        # self.assertAlmostEqual(r.chi, [-170.86489868164062, 59.921932220458984])
        # self.assertAlmostEqual(r.ca_location, {'x': 22.385000228881836, 'y': 17.197999954223633, 'z': 17.680999755859375})
        self.assertEqual(r.chain_break, chain_break_type.NoGap)
        self.assertEqual(r.nr, 15)
        self.assertEqual(r.type, structure_type.Alphahelix)
        self.assertEqual(r.ssBridgeNr, 0)
        self.assertEqual(r.helix(helix_type._3_10), helix_position_type.NoHelix)
        self.assertEqual(r.helix(helix_type.alpha), helix_position_type.Start)
        self.assertEqual(r.helix(helix_type.pi), helix_position_type.NoHelix)
        self.assertEqual(r.helix(helix_type.pp), helix_position_type.NoHelix)
        self.assertEqual(r.is_alpha_helix_end_before_start, False)
        self.assertEqual(r.bend, True)
        self.assertEqual(r.sheet, 0)
        self.assertEqual(r.strand, 0)

        (acceptor, energy) = r.acceptor(0)
        self.assertEqual(acceptor.seq_id, 17)
        self.assertAlmostEqual(energy, -0.2)
        
if __name__ == "__main__":
    unittest.main()