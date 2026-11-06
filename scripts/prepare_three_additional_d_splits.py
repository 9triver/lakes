"""D-style geographic holdouts for included Hubei/Guangdong/Shaanxi patches.

No imagery/labels are used to select splits; no neural network training.
"""
from __future__ import annotations
import csv
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from pyproj import Transformer
from shapely import segmentize
from shapely.geometry import box, mapping
from shapely.ops import transform, unary_union
from shapely.strtree import STRtree

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/additional_three_D_75_13_12_buffer2560_seed42_20261106'
PROVINCES=['hubei','guangdong','shaanxi']
SPLITS=['train','validation','test','excluded_buffer']
GAP=2560.
SEED=42
TARGET=np.array([.75,.13,.12])


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):
 with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def dump(path,data):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
def table(path,rows,fields=None):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 fields=fields or sorted(set().union(*(r.keys() for r in rows)))
 with path.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def note(**data):
 print(json.dumps(data,ensure_ascii=False),flush=True);dump(OUT/'status.json',data)


def footprints(rows):
 transformers={};result=[]
 for row in rows:
  g=box(*[float(row['bounds_'+n]) for n in ['left','bottom','right','top']])
  if row['crs']!='EPSG:32649':
   if row['crs'] not in transformers:transformers[row['crs']]=Transformer.from_crs(row['crs'],32649,always_xy=True)
   g=transform(transformers[row['crs']].transform,segmentize(g,128.))
  assert g.is_valid and not g.is_empty
  result.append(g)
 return result


def geographic_strata(lonlat,maximum):
 """Balanced geographic binary cells; never fewer than3 indivisible Sites/layer."""
 leaves=[np.arange(len(lonlat))]
 while len(leaves)<maximum:
  choices=[(np.ptp(lonlat[ids],axis=0).max(),i) for i,ids in enumerate(leaves) if len(ids)>=6]
  if not choices:break
  _,index=max(choices);ids=leaves.pop(index)
  # Longitude scaled by latitude; axis choice follows geographic extent only.
  xy=lonlat[ids].copy();xy[:,0]*=np.cos(np.deg2rad(lonlat[:,1].mean()))
  axis=int(np.argmax(np.ptp(xy,axis=0)))
  order=ids[np.argsort(xy[:,axis],kind='stable')]
  mid=len(order)//2;leaves.extend([order[:mid],order[mid:]])
 leaves.sort(key=lambda ids:tuple(lonlat[ids].mean(axis=0)))
 result=np.empty(len(lonlat),int)
 for i,ids in enumerate(leaves):result[ids]=i
 return result


def grid_candidates(bounds,strata,weights):
 result=[]
 for size in [30000.,40000.,50000.]:
  best=None
  for ox in np.arange(0,size,size/8):
   for oy in np.arange(0,size,size/8):
    cells=np.floor((bounds[:,:2]-[ox,oy])/size).astype(int);lo=cells*size+[ox,oy]
    eligible=np.all(bounds[:,:2]>=lo+GAP/2,axis=1)&np.all(bounds[:,2:]<=lo+size-GAP/2,axis=1)
    value=int(weights[eligible].sum())
    if best is None or value>best[0]:best=(value,float(ox),float(oy),cells,eligible)
  _,ox,oy,cells,eligible=best;buckets=defaultdict(list)
  for i in np.flatnonzero(eligible):buckets[int(cells[i,0]),int(cells[i,1]),int(strata[i])].append(int(i))
  keys=sorted(buckets);core=np.zeros((len(keys),len(weights)),bool)
  for j,key in enumerate(keys):core[j,buckets[key]]=True
  result.append(dict(size=size,ox=ox,oy=oy,keys=keys,core=core))
 return result


def search(grid,strata,weights,adjacency,blocked,seed):
 """Exhaust all small-province core allocations; seeded order breaks exact ties."""
 core=grid['core'];n=len(core);s=len(weights);layers=int(strata.max())+1
 if n<2:return None
 if n>15:raise ValueError('Unexpectedly many cores; explicit larger-search protocol required')
 rng=np.random.default_rng(seed);permutation=rng.permutation(n);core=core[permutation]
 powers=3**np.arange(n,dtype=np.int64);best=None
 for start in range(0,3**n,8192):
  codes=np.arange(start,min(start+8192,3**n),dtype=np.int64)
  labels=((codes[:,None]//powers)%3).astype('int8')
  val=(labels==1).astype('int16')@core.astype('int16')>0
  test=(labels==2).astype('int16')@core.astype('int16')>0
  compatible=~np.any((val.astype('int16')@adjacency.astype('int16')>0)&test,axis=1)
  labels,val,test,codes=labels[compatible],val[compatible],test[compatible],codes[compatible]
  if not len(codes):continue
  excluded=((val|test).astype('int16')@adjacency.astype('int16')>0)&~(val|test)
  assignment=np.where(val,1,np.where(test,2,np.where(excluded,3,0))).astype('int8')
  conflict=np.take_along_axis(np.broadcast_to(blocked,(len(codes),s,3)),assignment.clip(max=2)[:,:,None],axis=2).squeeze(2)
  assignment[(assignment<3)&conflict]=3
  counts=np.stack([(assignment==j)@weights for j in range(4)],axis=1)
  kept=counts[:,:3].sum(axis=1);fractions=counts[:,:3]/np.maximum(kept[:,None],1)
  regional=np.stack([np.stack([(assignment[:,strata==z]==j)@weights[strata==z] for j in range(3)],axis=1) for z in range(layers)],axis=1)
  feasible=np.all(regional>0,axis=(1,2)) & (np.max(np.abs(fractions-TARGET),axis=1)<.04)
  if not feasible.any():continue
  rf=regional/np.maximum(regional.sum(axis=2,keepdims=True),1)
  score=np.sum(((fractions-TARGET)/[.04,.025,.025])**2,axis=1)+np.mean(np.sum(((rf-TARGET)/[.12,.08,.08])**2,axis=2),axis=1)+4*counts[:,3]/weights.sum()
  score[~feasible]=np.inf;i=int(np.argmin(score))
  if best is None or float(score[i])<best['score']:
   original_labels=np.empty(n,'int8');original_labels[permutation]=labels[i]
   best=dict(score=float(score[i]),assignment=assignment[i].copy(),block_labels=original_labels,
             counts=counts[i].astype(int).tolist(),fractions=fractions[i].tolist(),regional_counts=regional[i].astype(int).tolist(),retention=float(kept[i]/weights.sum()))
 return best


def existing_references():
 snapshots=ROOT/'outputs/guizhou_hunan_chongqing_train_slope_kmeans_20261106/inputs'
 joint=ROOT/'outputs/guizhou_hunan_joint_k5_rgb_nir_slope_seed42_lr001_batch8_20261106/splits'
 extra=ROOT/'outputs/additional_three_D_reference_inputs_20261106'
 references={split:[] for split in SPLITS[:3]};protected={}
 for province in ['guizhou','hunan','chongqing']:
  path=snapshots/f'{province}_train.csv'
  protected[str(path)]=sha(path);references['train'].extend(footprints(read(path)))
 sources=json.loads((extra/'sources.json').read_text())
 for split in ['validation','test']:
  path=joint/f'{split}.csv'
  protected[str(path)]=sha(path);references[split].extend(footprints(read(path)))
  path=extra/f'chongqing_{split}.csv'
  assert sha(path)==sources[path.name]['geometry_sha256']
  protected[str(path)]=sha(path);references[split].extend(footprints(read(path)))
 return references,protected


def split_province(province,references):
 folder=OUT/province
 source=ROOT/f'data/workspaces/default/logical_patches/{province}/manifest.csv'
 original_hash=sha(source)
 with source.open(encoding='utf-8-sig',newline='') as f:
  rows=[r for r in csv.DictReader(f) if r.get('include','').lower()=='true']
 assert rows and all(r['region']==province for r in rows)
 for row in rows:row['patch_id']=row['logical_patch_id']
 assert len({r['patch_id'] for r in rows})==len(rows)
 grouped=defaultdict(list)
 for i,row in enumerate(rows):grouped[row['site_id']].append(i)
 site_ids=sorted(grouped);members=[grouped[k] for k in site_ids]
 geoms=footprints(rows);sites=[unary_union([geoms[i] for i in ids]) for ids in members]
 bounds=np.array([g.bounds for g in sites]);weights=np.array([len(v) for v in members])
 inverse=Transformer.from_crs(32649,4326,always_xy=True)
 lonlat=np.array([inverse.transform(g.centroid.x,g.centroid.y) for g in sites])
 adjacency=np.zeros((len(sites),len(sites)),bool)
 hits=STRtree(sites).query(sites,predicate='dwithin',distance=GAP)
 adjacency[hits[0],hits[1]]=True;np.fill_diagonal(adjacency,False)
 blocked=np.zeros((len(sites),3),bool);external_hits={}
 for split,reference in references.items():
  tree=STRtree(reference);pairs=tree.query(sites,predicate='dwithin',distance=GAP)
  near=np.zeros(len(sites),bool);near[pairs[0]]=True
  external_hits[split]=int(near.sum())
  for j,name in enumerate(SPLITS[:3]):
   if name!=split:blocked[:,j]|=near
 search_rows=[];winner=None
 for max_layers in range(min(4,len(sites)//3),0,-1):
  strata=geographic_strata(lonlat,max_layers)
  for number,grid in enumerate(grid_candidates(bounds,strata,weights)):
   best=search(grid,strata,weights,adjacency,blocked,SEED+number)
   record=dict(strata_count=int(strata.max())+1,block_size_m=grid['size'],origin_m=[grid['ox'],grid['oy']],feasible=best is not None)
   if best:
    record.update({k:best[k] for k in ['score','counts','fractions','regional_counts','retention']})
    if winner is None or best['score']<winner[0]['score']:winner=best,grid,strata
   search_rows.append(record);note(status='split_search',province=province,**record)
  if winner is not None:break
 if winner is None:
  dump(folder/'infeasibility.json',dict(site_counts=dict(zip(site_ids,weights.tolist())),search=search_rows,external_conflicts=external_hits))
  raise ValueError(f'{province}: cannot meet whole-Site/buffer/stratification/ratio constraints; no split frozen')
 best,grid,strata=winner;assignment=best['assignment']
 manifests={};assigned_rows={};checks={}
 for code,split in enumerate(SPLITS):
  selected=[]
  for i,ids in enumerate(members):
   if assignment[i]!=code:continue
   for j in ids:selected.append(dict(rows[j],spatial_split=split,spatial_group=f'D_{split}_{site_ids[i]}',geographic_stratum=f'geo_{strata[i]:02d}',
     exclusion_reason='Whole Site purged for2560m holdout buffer or conflict with a frozen other-province split' if code==3 else ''))
  assert selected or code==3
  path=folder/'splits'/('D_spatial_buffered/train.csv' if split=='train' else f'{split}.csv')
  output_fields=sorted(set(rows[0])|{'spatial_split','spatial_group','geographic_stratum','exclusion_reason'})
  table(path,selected,output_fields);assigned_rows[split]=selected
  manifests[split]=dict(path=str(path),sha256=sha(path),patches=len(selected),sites=len({r['site_id'] for r in selected}),fraction_of_source=len(selected)/len(rows),fraction_of_retained=len(selected)/sum(best['counts'][:3]) if code<3 else None)
  features=[dict(type='Feature',geometry=mapping(transform(inverse.transform,g)),properties=dict(site_id=site_ids[i],split=split,patches=int(weights[i]),geographic_stratum=f'geo_{strata[i]:02d}')) for i,g in enumerate(sites) if assignment[i]==code]
  dump(folder/f'maps/{split}_sites.geojson',dict(type='FeatureCollection',features=features))
 # Independent patch-level check after the site-level search.
 split_geoms={split:[geoms[j] for i,ids in enumerate(members) if assignment[i]==code for j in ids] for code,split in enumerate(SPLITS[:3])}
 for a,b in [('train','validation'),('train','test'),('validation','test')]:
  tree=STRtree(split_geoms[b]);hits=tree.query(split_geoms[a],predicate='dwithin',distance=GAP-1e-6)
  intersections=tree.query(split_geoms[a],predicate='intersects')
  minimum=unary_union(split_geoms[a]).distance(unary_union(split_geoms[b]))
  assert hits.shape[1]==intersections.shape[1]==0 and minimum>=GAP-1e-6
  checks[a+'__'+b]=dict(intersecting_patch_pairs=0,under2560m_pairs=0,minimum_distance_m=float(minimum))
 for new_split,gs in split_geoms.items():
  for old_split,old_gs in references.items():
   if new_split==old_split:continue
   pairs=STRtree(old_gs).query(gs,predicate='dwithin',distance=GAP-1e-6)
   assert pairs.shape[1]==0,(province,new_split,old_split,'cross-province leakage')
   checks[f'new_{new_split}__previous_{old_split}']=dict(under2560m_pairs=0)
 assert sum(v['patches'] for v in manifests.values())==len(rows)
 assert sha(source)==original_hash
 # Explicit membership and layers enable review without scanning all Patch rows.
 table(folder/'site_assignments.csv',[dict(site_id=sid,split=SPLITS[assignment[i]],patches=int(weights[i]),longitude=float(lonlat[i,0]),latitude=float(lonlat[i,1]),geographic_stratum=f'geo_{strata[i]:02d}') for i,sid in enumerate(site_ids)])
 dump(folder/'search_summary.json',search_rows)
 protocol=dict(region=province,seed=SEED,source_manifest=str(source),source_manifest_sha256=original_hash,
  source_filter='include=true only; excluded records unchanged',source_patches=len(rows),source_sites=len(sites),requested_fractions=dict(zip(SPLITS[:3],TARGET.tolist())),
  actual_fractions_are_of_retained_data=True,minimum_cross_split_distance_m=GAP,common_crs='EPSG:32649',
  geographic_stratification='Recursive geographic median splits along largest longitude*cos(latitude)/latitude extent; minimum3 Sites per layer; merge layers only if no feasible D-style allocation',
  strata_count=int(strata.max())+1,stratum_counts=best['regional_counts'],selected_grid=dict(block_size_m=grid['size'],origin_m=[grid['ox'],grid['oy']],holdout_core_inset_m=GAP/2),
  split_method='Hunan D-style whole-Site all-date union; geographic core nomination; validation/test distance constraints; whole-Site purging near holdouts; protect frozen other-province splits',
  selection='Exact enumeration of small-province core allocations; D score uses overall75:13:12, per-layer ratio error, retention; seeded core order breaks equal-score ties. No labels, imagery values, model scores used',
  score=best['score'],manifests=manifests,verification=checks,external_nearby_sites=external_hits,source_unchanged=True,training=False,
  limitations=['Only included Sites:12 Hubei/13 Guangdong/6 Shaanxi; limited geographic coverage, not province-representative',
   'Large Site patch weights and whole-Site constraints may prevent exact ratios; accepted deviation below4 percentage points',
   'Geographic stratification adapts Hunan latitude bands to sparse two-dimensional province geometry; numeric strata boundaries differ',
   'Existing historical models may have seen these locations; new splits do not make old model evaluations blind'])
 dump(folder/'protocol.json',protocol)
 for split in references:references[split].extend(split_geoms[split])
 return protocol


def main():
 if OUT.exists():
  if {p.name for p in OUT.iterdir()}=={'status.json'} and json.loads((OUT/'status.json').read_text()).get('status')=='failed':
   OUT.rename(OUT.with_name(OUT.name+f'_failed_preflight_{time.time_ns()}'))
  else:raise FileExistsError(f'Never overwrite an existing frozen split: {OUT}')
 OUT.mkdir(parents=True);start=time.time()
 references,protected=existing_references();results=[]
 for province in PROVINCES:
  result=split_province(province,references);results.append(result)
 assert all(sha(path)==digest for path,digest in protected.items())
 dump(OUT/'summary.json',dict(status='completed',regions=results,protected_original_manifests=protected,seconds=time.time()-start,training=False))
 note(status='completed',regions=PROVINCES,seconds=time.time()-start,training=False)


if __name__=='__main__':
 try:main()
 except Exception as exc:
  import traceback
  note(status='failed',error=str(exc),traceback=traceback.format_exc());raise
