from scboolseq import __version__
import scboolseq as scb
import pandas as pd

def test_version():
    assert __version__ == "0.2.0"

def test_single_bimodal_gene_data():    
    pd_data = pd.read_csv('./tests/input-data.csv', index_col=0)
    scboolseq = scb.scBoolSeq()
    scboolseq.fit(pd_data)
    pd_data_bin = scboolseq.binarize(pd_data)    
    assert set(pd_data_bin['Tcea1']) == {0.0, 1.0}    

def test_single_discarded_gene_data():    
    pd_data = pd.read_csv('./tests/input-data-2.csv', index_col=0)
    scboolseq = scb.scBoolSeq()
    scboolseq.fit(pd_data)
    pd_data_bin = scboolseq.binarize(pd_data)    
    print(scboolseq.criteria_)
    assert set(pd_data_bin['Tcea1']) == {}