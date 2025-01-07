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
    assert scboolseq.criteria_.iloc[0]['Category'] == 'Bimodal'
    assert set(pd_data_bin['Tcea1']) == {0.0, 1.0}    

def test_single_discarded_gene_data():    
    pd_data = pd.read_csv('./tests/input-data-2.csv', index_col=0)
    scboolseq = scb.scBoolSeq()
    scboolseq.fit(pd_data)
    pd_data_bin = scboolseq.binarize(pd_data)    
    assert scboolseq.criteria_.iloc[0]['Category'] == 'Discarded'
    assert 0.0 not in pd_data_bin['Tcea1']
    assert 1.0 not in pd_data_bin['Tcea1']

def test_problematic_KDE_bandwidth():    
    pd_data = pd.read_csv('./tests/input-data-3.csv', index_col=0)
    scboolseq = scb.scBoolSeq()
    scboolseq.fit(pd_data)
    pd_data_bin = scboolseq.binarize(pd_data)
    # This gets recognized as zero inflated, but otherwise is 
    # completely binarized as NaN.
    assert scboolseq.criteria_.iloc[0]['Category'] == 'ZeroInf'
    assert 0.0 not in pd_data_bin['Gm7293']
    assert 1.0 not in pd_data_bin['Gm7293']