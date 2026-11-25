from copy import deepcopy
import pytest
from research_framework.config import validate_config
from research_framework.experiments.plans import continuous_base,qwz_base,pilot_cases


def test_plan_all_cases_physically_validate():
    cases=pilot_cases()
    assert len({r['name'] for r in cases})==len(cases)
    assert {r['config']['experiment']['task'] for r in cases}=={'continuous_electrons','lattice_tfim','qwz_topology'}
    for row in cases:assert validate_config(row['config'])==row['config']


@pytest.mark.parametrize('change',[
    {'typo':{}}, {'embedding':{'unknown':True}}, {'experiment':{'seed':-1}},
    {'attention':{'window':{'mode':'trainable_per_head','shape':'hard','width':2.}}},
    {'wavefunction':{'type':'projector'}}, {'embedding':{'modes':[[.3,1]]}}])
def test_reject_invalid_before_model_creation(change):
    with pytest.raises(ValueError):validate_config(change)


def test_reject_covariance_without_transport_and_continuous_index_cutoff():
    c=qwz_base();c['attention']['type']='gauge_covariant'
    with pytest.raises(ValueError):validate_config(c)
    c=continuous_base();c['attention']['window'].update(shape='hard',mode='fixed')
    with pytest.raises(ValueError):validate_config(c)
