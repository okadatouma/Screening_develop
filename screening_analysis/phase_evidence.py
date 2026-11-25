"""Independent twist-family Chern diagnostic. Never labels zero-twist checkpoints.

python -m screening_analysis.phase_evidence MANIFEST --output NEW_DIRECTORY
Manifest: {"physical_system":"lambda-jellium", "basis_id":"...",
 "state_origin":"independently_optimized_twisted_ground_states",
 "grids":[{"path":"coarse.npz"},{"path":"fine.npz"}]}
NPZ: states[nx,ny,D], theta_x[nx], theta_y[ny], optionally sewing_x/y[D,D],
gap_lower_bounds[nx,ny]. States must share the SAME physical Hilbert basis.
Not the amplitudes on arbitrary shared walkers, and not HF one-particle Chern.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from screening_analysis.berry_validation import berry_links

SOURCE='https://arxiv.org/abs/cond-mat/0503172'


def evaluate_grid(path):
    with np.load(path,allow_pickle=False) as z:
        vector_input='states' in z
        if vector_input:
            states=z['states'];nx,ny=states.shape[:2]
        else:
            x=np.asarray(z['overlap_x'],complex);y=np.asarray(z['overlap_y'],complex)
            if x.ndim!=2 or x.shape!=y.shape or min(x.shape)<3 or not np.isfinite(x).all() or not np.isfinite(y).all():
                raise ValueError('finite same-shaped overlap_x/y[nx>=3,ny>=3] required')
            if 'overlap_includes_boundary_sewing' not in z or not bool(z['overlap_includes_boundary_sewing']):
                raise ValueError('overlap links must explicitly include physical boundary sewing')
            nx,ny=x.shape
        for name,count in [('theta_x',nx),('theta_y',ny)]:
            if name not in z or z[name].shape!=(count,) or not np.allclose(z[name],2*np.pi*np.arange(count)/count,atol=1e-12,rtol=0):
                raise ValueError('explicit uniform twist grid on [0,2pi) required')
        if vector_input:
            result=berry_links(states,**{key:z[key] for key in ('sewing_x','sewing_y') if key in z})
        else:
            minimum=float(min(abs(x).min(),abs(y).min()))
            if minimum<=1e-8:raise ValueError('vanishing twist overlap')
            ux=x/abs(x);uy=y/abs(y)
            curvature=np.angle(ux*np.roll(uy,-1,axis=0)*np.conj(np.roll(ux,-1,axis=1))*np.conj(uy))
            chern=float(curvature.sum()/(2*np.pi))
            result=dict(curvature=curvature,chern=chern,nearest_integer=int(np.rint(chern)),minimum_link_overlap=minimum,
                max_plaquette_phase=float(abs(curvature).max()),branch_margin=float(np.pi-abs(curvature).max()),
                interpretation='Chern of supplied many-body overlap links; twisted ground states and overlap errors require independent verification')
            for axis,link in [('x',x),('y',y)]:
                key='overlap_error_'+axis
                if key in z:
                    error=z[key]
                    if error.shape!=link.shape or not np.isfinite(error).all() or np.any(error<0):raise ValueError('nonnegative finite overlap-error arrays required')
                    positive=error>0
                    result['minimum_supplied_link_snr_'+axis]=float(np.min(abs(link[positive])/error[positive])) if positive.any() else None
        gaps=z['gap_lower_bounds'] if 'gap_lower_bounds' in z else None
        if gaps is not None and (gaps.shape!=(nx,ny) or not np.isfinite(gaps).all()):raise ValueError('gap grid shape/finiteness')
        result.update(grid_shape=[nx,ny],hilbert_dimension=states.shape[-1] if vector_input else None,
                      input_kind='common_basis_state_vectors' if vector_input else 'physical_manybody_overlap_links',
                      gap_lower_bound=float(gaps.min()) if gaps is not None else None,
                      source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        return result


def run(manifest_path,output):
    manifest=json.loads(manifest_path.read_text())
    if not manifest.get('basis_id'):raise ValueError('explicit shared physical basis_id required')
    grids=[];output.mkdir(parents=True,exist_ok=False)
    for index,entry in enumerate(manifest['grids']):
        path=manifest_path.parent/entry['path'];result=evaluate_grid(path)
        np.savez_compressed(output/f'grid-{index}-curvature.npz',curvature=result.pop('curvature'))
        grids.append(result)
    sizes=[min(g['grid_shape']) for g in grids]
    if len({g['hilbert_dimension'] for g in grids})>1:raise ValueError('refinement grids must share the same Hilbert basis dimension')
    refinement=len(grids)>=2 and len(set(sizes))==len(sizes) and max(sizes)>=1.5*min(sizes)
    stable=bool(grids) and len({g['nearest_integer'] for g in grids})==1
    gapped=bool(grids) and all(g['gap_lower_bound'] is not None and g['gap_lower_bound']>0 for g in grids)
    # These checks are necessary, not sufficient: state preparation and Monte
    # Carlo/variational/system-size convergence are not proven by state vectors.
    result=dict(grids=grids,chern_grid_stable=bool(refinement and stable),positive_supplied_gap_bounds=gapped,
        physical_system=manifest.get('physical_system','unspecified'),basis_id=manifest['basis_id'],
        state_origin=manifest.get('state_origin','unspecified'),physical_phase='unknown',phase_verified=False,
        missing_checks=['Independent provenance and convergence of twisted ground states',
                        'Thermodynamic crystal-order evidence for WC/AHC distinction',
                        'Independent Hall-response comparison if required'],
        note='Computes Chern of SUPPLIED state family. Fixed-walker samples are not an orthonormal Hilbert basis. No twist phase is appended to a single checkpoint.',
        method_source=SOURCE)
    (output/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('manifest',type=Path)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    result=run(args.manifest,args.output)
    print(json.dumps(dict(chern=[g['chern'] for g in result['grids']],grid_stable=result['chern_grid_stable'],phase_verified=False)))


if __name__=='__main__':main()
