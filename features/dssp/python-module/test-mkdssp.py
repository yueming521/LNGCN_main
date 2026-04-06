from mkdssp import dssp, helix_type, TestBond
import os
import gzip

file_path = os.path.join("..", "test", "1cbs.cif.gz")

with gzip.open(file_path, "rt") as f:
	file_content = f.read()
 
dssp = dssp(file_content)
 
print("residues: ", dssp.statistics.residues)
print("chains: ", dssp.statistics.chains)
print("SS_bridges: ", dssp.statistics.SS_bridges)
print("intra_chain_SS_bridges: ", dssp.statistics.intra_chain_SS_bridges)
print("H_bonds: ", dssp.statistics.H_bonds)
print("H_bonds_in_antiparallel_bridges: ", dssp.statistics.H_bonds_in_antiparallel_bridges)
print("H_bonds_in_parallel_bridges: ", dssp.statistics.H_bonds_in_parallel_bridges)

count = 0
for res in dssp:
	count += 1
	print(res.asym_id, res.seq_id, res.compound_id)
	print("alt_id", res.alt_id)
	print("compound_letter", res.compound_letter)
	print("auth_asym_id", res.auth_asym_id)
	print("auth_seq_id", res.auth_seq_id)
	print("pdb_strand_id", res.pdb_strand_id)
	print("pdb_seq_num", res.pdb_seq_num)
	print("pdb_ins_code", res.pdb_ins_code)
	print("alpha", res.alpha)
	print("kappa", res.kappa)
	print("phi", res.phi)
	print("psi", res.psi)
	print("tco", res.tco)
	print("omega", res.omega)
	print("is_pre_pro", res.is_pre_pro)
	print("is_cis", res.is_cis)
	print("chiral_volume", res.chiral_volume)
	print("chi", res.chi)
	print("ca_location", res.ca_location)
	print("chain_break", res.chain_break)
	print("nr", res.nr)
	print("type", res.type)
	print("ssBridgeNr", res.ssBridgeNr)
	print("helix(_3_10)", res.helix(helix_type._3_10))
	print("helix(alpha)", res.helix(helix_type.alpha))
	print("helix(pi)", res.helix(helix_type.pi))
	print("helix(pp)", res.helix(helix_type.pp))
	print("is_alpha_helix_end_before_start", res.is_alpha_helix_end_before_start)
	print("bend", res.bend)
	print("sheet", res.sheet)
	print("strand", res.strand)

	for i in range(0, 1):
		(ri, nr, dir) = res.bridge_partner(i)
		if ri != None:
			print("bridge partner ", i, ri.asym_id, ri.seq_id, ri.compound_id, nr, dir)

	for i in range(0, 1):
		(ri, e) = res.acceptor(i)
		if ri != None:
			print("acceptor ", i, ri.asym_id, ri.seq_id, ri.compound_id, e)
			print("test bond: ", TestBond(res, ri))
   	
	for i in range(0, 1):
		(ri, e) = res.donor(i)
		if ri != None:
			print("donor ", i, ri.asym_id, ri.seq_id, ri.compound_id, e)
			print("test bond: ", TestBond(res, ri))

	print("accessibility", res.accessibility)
	break


print("count: ", count)

a = dssp.get('A', 137)
b = dssp.get('A', 6)

print ("a & b", a, b)

assert(TestBond(a, b))
