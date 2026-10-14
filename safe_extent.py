"""Shared runtime/offline mask contract; face anchors do not prove limb identity.

Caller must separately review observed identity AND ownership before rendering.
Reject merged regions, keep established heads, protect ambiguous contacts, and
never repair an unknown person. No HTTP, models or case-specific coordinates.
"""
import numpy as np
from PIL import Image,ImageDraw,ImageFilter

def face_anchor_rois(faces,shape):
    """Scale-relative ROI hypotheses, not automatic body ownership labels."""
    if not 1<=len(faces)<=4 or any(f.shape!=shape for f in faces):raise ValueError('Invalid anchors/canvas')
    rows=[]
    for mode in ('shoulder','upper','full'):
        for index,f in enumerate(faces):
            y,x=np.where(f)
            if len(x)<100:raise ValueError('Empty anchor')
            x1,y1,x2,y2=int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)
            w,h=x2-x1,y2-y1;cx=(x1+x2)/2
            width={'shoulder':1.6,'upper':2.4,'full':3.0}[mode]*w
            bottom={'shoulder':y2+.8*h,'upper':y2+2.1*h,'full':shape[0]}[mode]
            rows.append({'mode':mode,'face_index':index,'face_bbox':[x1,y1,x2,y2], 'roi':[max(0,int(cx-width/2)),max(0,int(y1-.45*h)),min(shape[1],int(cx+width/2)),min(shape[0],int(bottom))]})
    return rows

def dilate(mask,radius):
    return np.asarray(Image.fromarray(mask.astype('uint8')*255).filter(ImageFilter.MaxFilter(2*radius+1)))>0

def inner_feather(mask,width=24):
    """Soft compositing weights strictly inside an already approved support.

    Does not expand ownership or authorize a previously excluded pixel.
    Eight-neighbour distance approximation; no scipy/runtime dependency.
    """
    if type(width) is not int or not 1<=width<=128:raise ValueError('Invalid feather width')
    remaining=mask.astype(bool).copy();distance=np.zeros(mask.shape,dtype=np.uint16)
    for _ in range(width):
        distance+=remaining
        p=np.pad(remaining,1,constant_values=False)
        remaining=p[:-2,:-2]&p[:-2,1:-1]&p[:-2,2:]&p[1:-1,:-2]&p[1:-1,1:-1]&p[1:-1,2:]&p[2:,:-2]&p[2:,1:-1]&p[2:,2:]
    x=distance.astype(np.float32)/width
    return np.rint(255*(x*x*(3-2*x))).astype('uint8')

def extent_masks(raw,faces,body,radius=24,allow_empty=False):
    if not 1<=len(raw)==len(faces)<=4:raise ValueError('Invalid counts')
    if type(radius) is not int or not 0<=radius<=128:raise ValueError('Invalid contact radius')
    if any(m.dtype!=np.bool_ for m in raw+faces+[body]):raise ValueError('Explicit binary support required')
    shape=body.shape
    if len(shape)!=2 or any(m.shape!=shape for m in raw+faces):raise ValueError('Canvas mismatch')
    regions=[];heads=np.zeros(shape,bool)
    for i,(m,f) in enumerate(zip(raw,faces)):
        ys,xs=np.where(f)
        if len(xs)<100:raise ValueError('Empty anchor')
        if any((m&other).any() for j,other in enumerate(faces) if j!=i):raise ValueError('Merged region includes another face')
        choices=np.argwhere(m&body&f)
        if len(choices)<100:raise ValueError('Own anchor not covered')
        cy,cx=(ys.min()+ys.max())/2,(xs.min()+xs.max())/2
        y,x=choices[np.argmin((choices[:,0]-cy)**2+(choices[:,1]-cx)**2)]
        im=Image.fromarray((m&body).astype('uint8')*255).copy();ImageDraw.floodfill(im,(int(x),int(y)),128,thresh=0)
        regions.append(np.asarray(im)==128)
        # Head protection is conservative geometry, NOT new identity evidence.
        w=xs.max()-xs.min()+1;h=ys.max()-ys.min()+1
        x1=max(0,int(xs.min()-.55*w));x2=min(shape[1],int(xs.max()+.55*w)+1)
        y1=max(0,int(ys.min()-.60*h));y2=min(shape[0],int(ys.max()+.18*h)+1)
        heads[y1:y2,x1:x2]=True
    contacts=np.zeros(shape,bool)
    for i in range(len(regions)):
        for j in range(i):contacts|=dilate(regions[i],radius)&dilate(regions[j],radius)
    protected=heads|dilate(contacts,8)
    safe=[m&~protected for m in regions]
    union=np.zeros(shape,bool)
    for m in safe:
        if m.sum()<100 and not allow_empty:raise ValueError('No meaningful safe extent; keep face fallback')
        if (m&union).any():raise ValueError('Unsafe overlap remains')
        union|=m
    return safe,union,protected,{'raw_overlap_pixels':int(sum((regions[i]&regions[j]).sum() for i in range(len(regions)) for j in range(i))), 'protected_pixels':int(protected.sum()), 'edited_pixels':int(union.sum()), 'ownership':'REQUIRES_OBSERVED_REVIEW_NOT_PROVEN_BY_FACE_COUNT', 'full_visible_person':'CONTACTS_AND_HEADS_EXCLUDED'}
