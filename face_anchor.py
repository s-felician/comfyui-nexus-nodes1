"""Face-anchored runtime bridge. Existing Impact/SAM performs segmentation.

No network/download/model loader/new ComfyUI class. Coordinates never identify
characters. The VLM must independently observe region ownership and identity.
"""
import base64
import hashlib
import json
import zlib
from collections import namedtuple
import numpy as np
import torch
from PIL import Image,ImageDraw
from .safe_extent import face_anchor_rois,extent_masks,dilate,inner_feather

SEG=namedtuple('SEG','cropped_image cropped_mask confidence crop_region bbox label control_net_wrapper')
FORMAT='NEXUS_FACE_ANCHORED_V1'

def image_hash(image):
    return hashlib.sha256(image.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()

def make_seg(mask,confidence=1.0):
    y,x=np.where(mask>0)
    if len(x)<1:raise ValueError('Empty region')
    box=(int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1));a,b,c,d=box
    return SEG(None,np.array(mask[b:d,a:c],dtype=np.float32,copy=True),float(confidence),box,box,'person',None)

def encode_mask(mask):
    if mask.dtype!=np.bool_:raise ValueError('Binary capsule required')
    return base64.b64encode(zlib.compress(np.packbits(mask.reshape(-1)).tobytes())).decode('ascii')

def decode_mask(value,shape):
    if not isinstance(value,str) or len(value)>8_000_000:raise ValueError('Invalid mask capsule')
    size=(shape[0]*shape[1]+7)//8;decoder=zlib.decompressobj()
    result=decoder.decompress(base64.b64decode(value,validate=True),size+1)
    if len(result)!=size or decoder.unconsumed_tail or decoder.unused_data or not decoder.eof:raise ValueError('Mask capsule size/stream mismatch')
    return np.unpackbits(np.frombuffer(result,dtype=np.uint8),count=shape[0]*shape[1]).reshape(shape).astype(bool)

def seg_hash(seg,shape):
    from . import _seg_mask
    m=_seg_mask(seg,*shape,include_soft_support=True).numpy()
    metadata=json.dumps({'bbox':list(map(float,seg.bbox)),'crop':list(map(int,seg.crop_region))},sort_keys=True).encode()
    return hashlib.sha256(metadata+np.packbits(m).tobytes()).hexdigest()

def sam_mask(model,segs,image):
    from impact.core import make_sam_mask
    return make_sam_mask(model,segs,image,'center-1',0,0.93,0,0.7,'False')

def binary_result(value,shape,strict=False):
    a=torch.as_tensor(value).detach().float().cpu()
    if a.ndim==3 and a.shape[0]==1:a=a[0]
    if tuple(a.shape)!=shape or not torch.isfinite(a).all() or (a<0).any() or (a>1).any():raise ValueError('Invalid actual mask result')
    if strict and not torch.all((a==0)|(a==1)):raise ValueError('Existing SAM binary-mask contract changed')
    return a.numpy()>0

def component(mask,face,body):
    options=np.argwhere(mask&face&body)
    if len(options)<100:raise ValueError('SAM misses its observed face')
    y,x=np.where(face);cy,cx=(y.min()+y.max())/2,(x.min()+x.max())/2
    sy,sx=options[np.argmin((options[:,0]-cy)**2+(options[:,1]-cx)**2)]
    im=Image.fromarray((mask&body).astype('uint8')*255).copy();ImageDraw.floodfill(im,(int(sx),int(sy)),128,thresh=0)
    return np.asarray(im)==128

def raw_support(segs,shape):
    from . import _seg_mask
    if not isinstance(segs,(tuple,list)) or len(segs)!=2 or tuple(segs[0])!=shape or len(segs[1])>12:raise ValueError('Detector canvas/count mismatch')
    out=np.zeros(shape,bool)
    for seg in segs[1]:out|=_seg_mask(seg,*shape,include_soft_support=True).numpy()
    return out

def face_bbox_support(seg,shape):
    from . import _seg_mask
    _seg_mask(seg,*shape)  # Validate crop_region independently, never use it as bbox.
    a,b,c,d=seg.bbox
    out=np.zeros(shape,bool)
    out[int(np.floor(b)):int(np.ceil(d)),int(np.floor(a)):int(np.ceil(c))]=True
    return out

def detail_guard(image,faces,body):
    """Conservative chest/contrast protection; not garment recognition."""
    rgb=image[0].detach().float().cpu().numpy();edges=np.zeros(body.shape,bool)
    edges[1:]|=np.max(np.abs(rgb[1:]-rgb[:-1]),axis=2)>.18
    edges[:,1:]|=np.max(np.abs(rgb[:,1:]-rgb[:,:-1]),axis=2)>.18
    protected=dilate(edges&body,4)
    for f in faces:
        y,x=np.where(f);w=x.max()-x.min()+1;h=y.max()-y.min()+1
        a=max(0,int(x.min()-.55*w));c=min(body.shape[1],int(x.max()+.55*w)+1)
        b=max(0,int(y.min()-.60*h));d=min(body.shape[0],int(y.max()+.65*h)+1)
        protected[b:d,a:c]=True
    return protected

def prepare_anchored(image,segs,face_segs,hand_segs,sam_model,protect_details=True,manual_detail_mask=None):
    from . import _seg_mask,_overlay,_ordered_segs
    if not isinstance(image,torch.Tensor) or image.ndim!=4 or image.shape[0]!=1 or image.shape[-1]!=3:raise ValueError('One actual RGB IMAGE required')
    if not torch.isfinite(image).all() or (image<0).any() or (image>1).any():raise ValueError('Invalid IMAGE')
    shape=tuple(map(int,image.shape[1:3]))
    if shape[0]*shape[1]>32_000_000:raise ValueError('Canvas too large')
    actual_body=raw_support(segs,shape);body=actual_body.copy();hands=dilate(raw_support(hand_segs,shape),24)
    if tuple(face_segs[0])!=shape or not 1<=len(face_segs[1])<=4:raise ValueError('1..4 real face anchors required')
    faces=[face_bbox_support(s,shape) for s in face_segs[1]]
    if any(int(f.sum())<100 for f in faces):raise ValueError('Tiny face anchor')
    if any((faces[i]&faces[j]).any() for i in range(len(faces)) for j in range(i)):raise ValueError('Overlapping/duplicate face anchors')
    unknown_body=np.zeros(shape,bool);unknown_count=0
    from . import _seg_mask
    for seg in segs[1]:
        support=_seg_mask(seg,*shape,include_soft_support=True).numpy()>0
        if support.any() and not any((support&f).any() for f in faces):unknown_body|=support;unknown_count+=1
    face_only=not body.any()
    if face_only:
        for f in faces:body|=f
    rois=[r for r in face_anchor_rois(faces,shape) if r['mode']=='full']
    regions=[];face_masks=[];skipped=[];body_rejected=[];face_rejected=[];keep=[]
    for i,row in enumerate(rois):
        try:
            if not face_only:
                roi=np.zeros(shape,bool);a,b,c,d=row['roi'];roi[b:d,a:c]=True
                try:
                    raw=binary_result(sam_mask(sam_model,(shape,[make_seg(roi)]),image),shape,strict=True)
                    region=component(raw,faces[i],body)
                    if any((region&other).any() for j,other in enumerate(faces) if j!=i):raise ValueError('Merged SAM region contains another face')
                except ValueError as exc:
                    body_rejected.append({'face_index':i,'reason':str(exc)[:120]});region=faces[i]&body
            else:region=faces[i]&body
            try:
                face_raw=binary_result(sam_mask(sam_model,(shape,[face_segs[1][i]]),image),shape,strict=True)
                if any((face_raw&other).any() for j,other in enumerate(faces) if j!=i):raise ValueError('Face SAM includes another face')
                face=component(face_raw,faces[i],body)&faces[i]&~hands
                if manual_detail_mask is not None:face&=~binary_result(manual_detail_mask,shape)
                if face.sum()<100:raise ValueError('No safely editable face')
            except ValueError as exc:
                face_rejected.append({'face_index':i,'reason':str(exc)[:120]});face=np.zeros(shape,bool)
            if region.sum()<100:raise ValueError('No observed instance support')
            # Body SAM and face SAM are independent observations. The body
            # candidate may omit parts of its own safely verified face. Keep
            # the face in this observation, without extending either edit mask.
            regions.append(region|face);face_masks.append(face);keep.append(i)
        except ValueError as exc:skipped.append({'face_index':i,'reason':str(exc)[:120]})
    if not regions:raise ValueError('Zero safe observed instances; VLM/CatFlat blocked')
    _,_,protected,_=extent_masks(regions,[faces[i] for i in keep],body,allow_empty=True)
    protected|=hands
    protected|=dilate(unknown_body,24)
    for j,f in enumerate(faces):
        if j not in keep:protected|=dilate(f,24)
    if protect_details:protected|=detail_guard(image,faces,body)
    # A rejected face-anchored instance means unknown limb ownership in a raw
    # merged support. Keep known faces, but do not paint anyone's body yet.
    if skipped or body_rejected or face_only:protected|=body
    if manual_detail_mask is not None:protected|=binary_result(manual_detail_mask,shape)
    contact=np.zeros(shape,bool)
    for i in range(len(regions)):
        for j in range(i):contact|=dilate(regions[i],24)&dilate(regions[j],24)
    obs=[m&~dilate(contact,8) for m in regions]
    candidate=(shape,[make_seg(m,getattr(face_segs[1][keep[i]],'confidence',1.0)) for i,m in enumerate(obs)]);_,safe,rejected=_ordered_segs(candidate,image)
    if len(safe)!=len(regions):raise ValueError('Ambiguous observed instance order')
    keys={seg_hash(s,shape):i for i,s in enumerate(candidate[1])};rows=[]
    for p,row in enumerate(safe):
        i=keys[seg_hash(row['seg'],shape)];edit=regions[i]&~protected;face=face_masks[i]&~dilate(contact,8)
        if face.sum()<100:face=np.zeros(shape,bool)
        rows.append({'instance_index':p,'detector_face_index':keep[i],'observation_hash':seg_hash(row['seg'],shape),
                     'edit_mask':encode_mask(edit),'face_mask':encode_mask(face),
                     'face_bbox':list(map(float,face_segs[1][keep[i]].bbox)),
                     'body_edit_pixels':int(edit.sum()),'face_edit_pixels':int(face.sum())})
    if not any(r['body_edit_pixels'] or r['face_edit_pixels'] for r in rows):raise ValueError('Zero safely editable regions; VLM/CatFlat blocked')
    meta={'format':FORMAT,'shape':list(shape),'image_sha256':image_hash(image),'safe_indices':list(range(len(rows))),
          'detected_face_count':len(faces),
          'unassigned_raw_person_regions':unknown_count,
          'original_detector_indices':[r['detector_face_index'] for r in rows],'skipped':skipped+rejected,'rows':rows,
          'body_rejected':body_rejected,'face_rejected':face_rejected,
          'protected_mask':encode_mask(protected),'hand_mask':encode_mask(hands),
          'raw_body_hash':hashlib.sha256(np.packbits(actual_body).tobytes()).hexdigest(),
          'person_extent':'FACE_ONLY_NO_BODY_DETECTION' if face_only else 'SAFE_VISIBLE_BODY_CANDIDATE',
          'ownership':'VLM_OBSERVED_OWNERSHIP_REQUIRED_NOT_PROVEN_BY_FACE_COUNT',
          'detail_protection':'CONSERVATIVE_CHEST_AND_HIGH_CONTRAST_EDGES' if protect_details else 'DISABLED',
          'sam':'EXISTING_IMPACT_CENTER_1_FALSE_NO_DOWNLOAD','new_classes':0,'VLM_requests':0}
    return _overlay(image,safe,[r['face_bbox'] for r in rows]),(shape,[r['seg'] for r in safe]),json.dumps(meta,ensure_ascii=False,separators=(',',':'))

def capsule(report,segs,image):
    from . import _strict_json,_seg_mask
    if not isinstance(report,str) or len(report)>8_000_000:raise ValueError('Invalid geometry capsule')
    meta=_strict_json(report)
    if meta.get('format')!=FORMAT or image is None or meta.get('image_sha256')!=image_hash(image):raise ValueError('Geometry/source IMAGE mismatch')
    shape=tuple(map(int,segs[0]))
    if tuple(image.shape[1:3])!=shape or shape[0]*shape[1]>32_000_000:raise ValueError('Geometry/source canvas mismatch')
    if meta.get('shape')!=list(shape) or len(meta.get('rows',[]))!=len(segs[1]):raise ValueError('Geometry shape/count mismatch')
    protected=decode_mask(meta['protected_mask'],shape);hands=decode_mask(meta['hand_mask'],shape)
    rows=[];seen=np.zeros(shape,bool);face_union=np.zeros(shape,bool)
    for i,(row,seg) in enumerate(zip(meta['rows'],segs[1])):
        if type(row.get('instance_index')) is not int or row['instance_index']!=i or row.get('observation_hash')!=seg_hash(seg,shape):raise ValueError('P-index geometry changed after observation')
        edit=decode_mask(row['edit_mask'],shape);face=decode_mask(row['face_mask'],shape);obs=_seg_mask(seg,*shape,include_soft_support=True).numpy()
        if (edit&~obs).any() or (edit&protected).any() or (edit&hands).any() or (edit&seen).any():raise ValueError('Unsafe body mask capsule')
        if 0<face.sum()<100 or (face&~obs).any() or (face&hands).any() or (face&face_union).any():raise ValueError('Unsafe face mask capsule')
        box=row.get('face_bbox')
        if not isinstance(box,list) or len(box)!=4 or any(isinstance(x,bool) or not isinstance(x,(float,int)) or not np.isfinite(x) for x in box):raise ValueError('Invalid face coordinates')
        a,b,c,d=box
        if not 0<=a<c<=shape[1] or not 0<=b<d<=shape[0]:raise ValueError('Face bbox outside image')
        facebox=np.zeros(shape,bool);facebox[int(b):int(np.ceil(d)),int(a):int(np.ceil(c))]=True
        if (face&~facebox).any():raise ValueError('Face mask outside observed face bbox')
        seen|=edit;face_union|=face;rows.append({**row,'edit':edit,'face':face})
    return meta,rows,protected

def render_set(pairs,rows,shape,cast,planner,scene,accepted_slots,kind):
    from . import _person_prompt,_reject_impact_control_syntax
    entries=[]
    for index,slot in pairs:
        mask=rows[index][kind]
        if mask.sum()<100:continue
        seg=make_seg(mask);prompt=_person_prompt(slot,cast,planner,scene,accepted_slots);_reject_impact_control_syntax(prompt)
        entries.append((seg,index,slot,prompt,mask))
    entries.sort(key=lambda x:(x[0].bbox[0],x[0].bbox[1]))
    if any(entries[i][0].bbox[:2]==entries[i-1][0].bbox[:2] for i in range(1,len(entries))):raise ValueError('Ambiguous render sort')
    wildcard='[ASC]'+'[SEP]'.join(x[3] for x in entries);union=np.zeros(shape,bool);audit=[]
    for seg,index,slot,prompt,mask in entries:
        if (union&mask).any():raise ValueError('Render masks overlap')
        union|=mask;audit.append({'instance_index':index,'slot':slot,'character_id':cast[slot].split(',')[1].strip(),
          'mask_sha256':hashlib.sha256(np.packbits(mask).tobytes()).hexdigest(),'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest()})
    return (shape,[x[0] for x in entries]),wildcard,union,audit
