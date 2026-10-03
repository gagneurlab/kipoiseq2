import pytest

vcf_file = "tests/data/test.vcf.gz"
test_with_multiple_variants = "tests/data/test_with_multiple_variants.vcf.gz"
sample_5kb_fasta_file = "tests/data/sample.5kb.fa"

# A plain-text VCF with the cases that a VCF reader has to handle:
# multi-allelic records, symbolic and `*` ALTs, ALT `.`, `N` ALTs,
# and haploid, missing and partly missing genotypes.
# The columns of the records are separated by spaces here and by tabs in the file.
_EDGE_CASE_HEADER = """\
##fileformat=VCFv4.2
##contig=<ID=chr1,length=1000>
##contig=<ID=chr2,length=1000>
##ALT=<ID=DEL,Description="Deletion">
##INFO=<ID=END,Number=1,Type=Integer,Description="End position">
##INFO=<ID=DP,Number=1,Type=Integer,Description="Total depth">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##FORMAT=<ID=GQ,Number=1,Type=Integer,Description="Genotype quality">
##FORMAT=<ID=AD,Number=R,Type=Integer,Description="Allelic depths">
"""
_EDGE_CASE_RECORDS = """\
#CHROM POS ID REF ALT QUAL FILTER INFO FORMAT S1 S2 S3
chr1 10 rs1 A C,G 50 PASS DP=10 GT:GQ:AD 0/2:30:5,0,5 1|1:20:0,8,0 ./.:.:.
chr1 20 . ACGT <DEL>,* . . END=40 GT:GQ:AD 0/1:10:3,3,0 1/2:10:0,2,2 0/0:10:6,0,0
chr1 30 . T . . . . GT:GQ:AD 0/0:10:4 0:10:4 .:.:.
chr1 40 . G N . . . GT:GQ:AD 0/1:10:2,2 0/0:10:4,0 0/0:10:4,0
chr2 5 . TA T,TAA,N . . . GT:GQ:AD 1:10:0,3,0,0 3/2:10:0,0,2,2 .|1:10:.
"""
EDGE_CASE_VCF = _EDGE_CASE_HEADER + "".join("\t".join(line.split()) + "\n" for line in _EDGE_CASE_RECORDS.splitlines())


@pytest.fixture
def edge_case_vcf(tmp_path):
    """Path of EDGE_CASE_VCF written to a temporary directory."""
    path = tmp_path / "edge_cases.vcf"
    path.write_text(EDGE_CASE_VCF)
    return str(path)
