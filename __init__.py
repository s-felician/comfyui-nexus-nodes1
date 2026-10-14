"""Small fail-closed SEGS/identity bridge. Detection and drawing use Impact Pack.

Two ComfyUI nodes in one package: observer overlay, verified SEGS/prompt binding.
No network, model loading or diffusion code lives here. Face-anchored mode
delegates segmentation to the installed Impact/SAM implementation.
"""
from __future__ import annotations

from collections import Counter
import json
import re

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


def _strict_json(value):
    if not isinstance(value, str):
        raise ValueError("VLM output must be a JSON string")
    if value.lstrip().startswith(chr(96)*3):
        raise ValueError("VLM response must not contain Markdown fences")
    def no_duplicate(pairs):
        result = {}
        for key, item in pairs:
            if key in result:
                raise ValueError("duplicate JSON key: " + key)
            result[key] = item
        return result
    def no_constant(item):raise ValueError("non-finite JSON constant: "+item)
    result = json.loads(value, object_pairs_hook=no_duplicate,parse_constant=no_constant)
    if not isinstance(result, dict):
        raise ValueError("VLM response must be a JSON object")
    return result


def _seg_mask(seg, height, width, *, include_soft_support=False):
    crop = getattr(seg, "crop_region", None)
    box = getattr(seg, "bbox", None)
    if crop is None or box is None or len(crop) != 4 or len(box) != 4:
        raise ValueError("missing bbox or crop_region")
    if any(not isinstance(x, (int, np.integer)) for x in crop):
        raise ValueError("Impact crop_region must use integer pixels")
    if any(not isinstance(x, (int, float, np.integer, np.floating)) or not np.isfinite(float(x)) for x in box):
        raise ValueError("Impact bbox must contain finite numeric coordinates")
    x0, y0, x1, y1 = map(int, crop)
    bx0, by0, bx1, by1 = map(float, box)
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError("crop outside image")
    if not (0 <= bx0 < bx1 <= width and 0 <= by0 < by1 <= height):
        raise ValueError("bbox outside image")
    raw = torch.as_tensor(seg.cropped_mask, dtype=torch.float32, device="cpu")
    if raw.ndim == 3 and raw.shape[0] == 1:
        raw = raw[0]
    if raw.ndim != 2 or tuple(raw.shape) != (y1-y0, x1-x0):
        raise ValueError("cropped mask dimensions differ from crop_region")
    if not torch.isfinite(raw).all():
        raise ValueError("mask has NaN/Inf")
    if (raw < 0).any() or (raw > 1).any():
        raise ValueError("mask values outside [0,1]")
    mask = torch.zeros((height, width), dtype=torch.bool)
    # Impact Detailer checks raw non-zero mask support before consuming its
    # wildcard chooser, while the verified composite uses a binary >=0.5 mask.
    # Use *all* non-zero support when detecting cross-person interference.
    mask[y0:y1, x0:x1] = (raw > 0) if include_soft_support else (raw >= 0.5)
    return mask


def _ordered_segs(segs, image):
    if not isinstance(image, torch.Tensor) or image.ndim != 4 or tuple(image.shape[:1]) != (1,) or image.shape[-1] != 3:
        raise ValueError("Only one RGB IMAGE supported")
    if not image.is_floating_point() or not torch.isfinite(image).all():
        raise ValueError("IMAGE must contain finite floating-point pixels")
    h, w = map(int, image.shape[1:3])
    if not isinstance(segs, (list, tuple)) or len(segs) != 2 or tuple(segs[0]) != (h,w):
        raise ValueError("SEGS header does not match Hires IMAGE")
    raw_items = list(segs[1])
    if len(raw_items) > 12:
        raise ValueError("too many detected regions; manual review required")
    candidates, rejected = [], []
    for index, seg in enumerate(raw_items):
        if str(getattr(seg,"label", "")).lower() != "person":
            rejected.append({"detector_index":index,"reason":"NON_PERSON"})
            continue
        try:
            mask = _seg_mask(seg,h,w)
            support = _seg_mask(seg,h,w,include_soft_support=True)
        except (ValueError,TypeError,AttributeError,RuntimeError) as exc:
            rejected.append({"detector_index":index,"reason":"INVALID_MASK","detail":str(exc)[:90]})
            continue
        area = int(mask.sum().item())
        if area < 100:
            rejected.append({"detector_index":index,"reason":"EMPTY_OR_TINY_MASK"})
            continue
        candidates.append({"detector_index":index,"seg":seg,"mask":mask,"support":support,"area":area})
    excluded = set()
    for i,a in enumerate(candidates):
        for j in range(i+1,len(candidates)):
            b = candidates[j]
            if (a["support"] & b["support"]).any():
                excluded.update((i,j))
    for i in sorted(excluded):
        rejected.append({"detector_index":candidates[i]["detector_index"],"reason":"OVERLAPPING_PERSON_MASK"})
    safe = [r for i,r in enumerate(candidates) if i not in excluded]
    # Match upstream DetailerForEach [ASC] precisely, not crop_region order.
    safe.sort(key=lambda r:(r["seg"].bbox[0],r["seg"].bbox[1]))
    for i in range(1,len(safe)):
        if tuple(safe[i-1]["seg"].bbox[:2]) == tuple(safe[i]["seg"].bbox[:2]):
            raise ValueError("identical bbox sort coordinates cannot establish stable instance order")
    if len(safe)>4:
        raise ValueError("more than four safe people; cannot infer original cast")
    return (h,w),safe,rejected


def _assert_impact_asc_safe(segs):
    """Validate exactly what upstream DetailerForEach will consume.

    VLM P-index must still correspond to sorted non-empty, non-overlapping
    SEGS when the final binder runs. A changed/reordered SEGS must fail rather
    than silently assigning the right prompt to the wrong mask.
    """
    if not isinstance(segs,(tuple,list)) or len(segs)!=2:
        raise ValueError("unsafe ASC SEGS container")
    header,regions=segs
    if (not isinstance(header,(tuple,list)) or len(header)!=2
        or any(type(v)!=int or v<=0 for v in header)):
        raise ValueError("unsafe ASC SEGS dimensions")
    height,width=header
    if not isinstance(regions,(tuple,list)) or not 1<=len(regions)<=4:
        raise ValueError("unsafe ASC SEGS person count")
    prev_key=None
    supports=[]
    for index,seg in enumerate(regions):
        if str(getattr(seg,"label","")).lower()!="person":
            raise ValueError("unsafe ASC non-person region")
        hard=_seg_mask(seg,height,width)
        if int(hard.sum().item())<100:
            raise ValueError("unsafe ASC empty or tiny person mask")
        support=_seg_mask(seg,height,width,include_soft_support=True)
        key=(float(seg.bbox[0]),float(seg.bbox[1]))
        if prev_key is not None and key<=prev_key:
            raise ValueError("unsafe ASC order or ambiguous bbox position")
        if any((support & earlier).any().item() for earlier in supports):
            raise ValueError("unsafe ASC overlapping person masks")
        supports.append(support)
        prev_key=key


def _overlay(image,safe,face_boxes=None):
    rgb = (image[0].detach().float().cpu().clamp(0,1).numpy()*255).round().astype(np.uint8)
    original=Image.fromarray(rgb,"RGB")
    width,height=original.size
    scale=min(1.0, 1200/max(width,1))
    global_img=original.resize((round(width*scale),round(height*scale)),Image.Resampling.LANCZOS)
    panel_w=700 if face_boxes is not None else 440
    cell_h=285 if face_boxes is not None else 255
    canvas=Image.new("RGB",(global_img.width+panel_w,max(global_img.height,cell_h*max(1,len(safe)))),(38,38,38))
    canvas.paste(global_img,(0,0))
    draw=ImageDraw.Draw(canvas)
    try: font=ImageFont.truetype("DejaVuSans.ttf",23)
    except OSError: font=ImageFont.load_default()
    for i,row in enumerate(safe):
        x0,y0,x1,y1=map(int,row["seg"].bbox)
        p=(round(x0*scale),round(y0*scale),round(x1*scale),round(y1*scale))
        draw.rectangle(p,outline=(255,225,10),width=5)
        draw.text((p[0]+4,p[1]+4),f"P{i}",font=font,fill=(255,255,0),stroke_width=3,stroke_fill=(0,0,0))
        cx0,cy0,cx1,cy1=map(int,row["seg"].crop_region)
        crop=original.crop((cx0,cy0,cx1,cy1))
        # Impact crop_region is rectangular and may contain neighboring actors.
        # Display only this instance's own segmentation pixels to the VLM.
        own_mask=(row["mask"][cy0:cy1,cx0:cx1].numpy().astype(np.uint8)*255)
        isolated=Image.new("RGB",crop.size,(38,38,38))
        isolated.paste(crop,(0,0),Image.fromarray(own_mask,"L"))
        crop=isolated
        ratio=min((330 if face_boxes is not None else 410)/max(crop.width,1),205/max(crop.height,1))
        crop=crop.resize((max(1,round(crop.width*ratio)),max(1,round(crop.height*ratio))),Image.Resampling.LANCZOS)
        x=global_img.width+15; y=i*cell_h+30
        canvas.paste(crop,(x,y))
        draw.text((x,i*cell_h+2),f"P{i}: exact SEGS crop",font=font,fill=(255,225,10))
        if face_boxes is not None:
            a,b,c,d=map(int,face_boxes[i]);face_crop=original.crop((a,b,c,d))
            face_crop.thumbnail((320,205),Image.Resampling.LANCZOS)
            canvas.paste(face_crop,(x+345,y))
            draw.text((x+345,i*cell_h+2),f"P{i}: observed face",font=font,fill=(70,240,255))
    return torch.from_numpy(np.asarray(canvas).copy().astype(np.float32)/255.0).unsqueeze(0)


def _features(tags):
    items=[x.strip().lower().replace(" ","_") for x in tags.split(",") if x.strip()]
    if len(items)<2:
        raise ValueError("canonical cast has no series/character_id")
    clothes=("_blazer","_uniform","_dress","_skirt","_shirt","_coat","_jacket","_hoodie","_kimono","_pants","_suit","_sweater","_tie","_clothes","_sleeves","_sash","_hakama")
    return {
        "character_id":items[1],
        "hair":{t for t in items if t.endswith("_hair") and t not in ("long_hair","short_hair","medium_hair","wavy_hair","straight_hair","curly_hair")},
        "eyes":{t for t in items if t.endswith("_eyes")},
        "accessory":{t for t in items if ("hairclip" in t or "headband" in t or "hair_ornament" in t or "hair_ribbon" in t or "hair_bow" in t or t in ("glasses","headphones","hairband","hat","necklace","earrings","brooch"))},
        "outfit":{t for t in items if t.endswith(clothes) or t in ("obi","hakama","kimono")}
    }


def _identity_ledger(catalogue,identities,active_count):
    """Resolve original dictionary IDs from selected identity, never geometry/tags."""
    if type(active_count)!=int or not 1<=active_count<=4:raise ValueError("Invalid active identity count")
    catalog=_strict_json(catalogue)
    def tokens(text):
        if not isinstance(text,str):raise ValueError("Invalid catalogue identity")
        return tuple(x.strip().casefold() for x in text.split(",") if x.strip())
    entries={key:tokens(value) for key,value in catalog.items()}
    result=[]
    import hashlib
    for slot,text in enumerate(identities[:active_count],1):
        selected=tokens(text)
        matches=[key for key,value in entries.items() if value==selected]
        if not matches and len(selected)>=2:
            matches=[key for key,value in entries.items() if value[:2]==selected[:2]]
        if len(matches)!=1:raise ValueError("Selected identity is absent/ambiguous in original catalogue")
        key=matches[0]
        if not isinstance(key,str) or not key.strip():raise ValueError("Invalid logical character ID")
        result.append({"slot":slot,"character_id":key,"identity_tokens":list(selected),
            "identity_sha256":hashlib.sha256(text.encode()).hexdigest()})
    if len({r["character_id"] for r in result})!=active_count:raise ValueError("Duplicate original logical ID")
    return json.dumps({"format":"NEXUS_LOGICAL_ID_V1","active_count":active_count,"characters":result},ensure_ascii=False)


def _ledger_ids(ledger,cast):
    obj=_strict_json(ledger)
    rows=obj.get("characters")
    if obj.get("format")!="NEXUS_LOGICAL_ID_V1" or obj.get("active_count")!=len(cast) or not isinstance(rows,list) or len(rows)!=len(cast):raise ValueError("Invalid logical identity ledger")
    ids={}
    for slot,row in enumerate(rows,1):
        if row.get("slot")!=slot or not isinstance(row.get("character_id"),str) or not row["character_id"]:raise ValueError("Invalid ledger slot")
        selected=row.get("identity_tokens")
        actual=[x.strip().casefold() for x in cast[slot].split(",") if x.strip()]
        if not isinstance(selected,list) or not selected or actual[:len(selected)]!=selected:raise ValueError("Identity/Outfit and logical ledger diverged")
        ids[slot]=row["character_id"]
    if len(set(ids.values()))!=len(ids):raise ValueError("Duplicate ledger identity")
    return ids


def _matches(record, slot, cast,character_ids=None):
    traits={s:_features(v) for s,v in cast.items()}
    expected=traits[slot]
    if record.get("character_id")!=(character_ids or {}).get(slot,expected["character_id"]) or record.get("ambiguous") is not False:
        return False
    obs=record.get("observed_features")
    if not isinstance(obs,dict):return False
    observed={}
    for key in ("hair","eyes","accessory","outfit"):
        value=obs.get(key,[])
        if not isinstance(value,list) or any(not isinstance(x,str) for x in value):return False
        observed[key]={x.strip().lower().replace(" ","_") for x in value if x.strip().lower() not in ("unknown","none","")}
        if observed[key] and not observed[key].issubset(expected[key]):return False
    if len([k for k,v in observed.items() if v])<2:return False
    if len(expected["eyes"])>1 and observed["eyes"]!=expected["eyes"]:return False
    candidates=[s for s,t in traits.items() if all(v.issubset(t[key]) for key,v in observed.items())]
    return candidates==[slot]


def _person_prompt(slot,cast,planner,scene,confirmed_slots):
    traits=cast[slot].strip()
    fields=("role","action","expression","gaze","pose","position")
    required={}
    for key in fields:
        value=planner.get(f"slot{slot}_{key}","")
        if not isinstance(value,str) or not value.strip():
            raise ValueError(f"Planner slot{slot}_{key} missing")
        required[key]=value.strip()
    relations=[]
    active={f"slot{s}" for s in cast}
    confirmed={f"slot{s}" for s in confirmed_slots}
    for rel in planner.get("relations",[]):
        if not isinstance(rel,dict):raise ValueError("invalid relation")
        source,target=rel.get("source"),rel.get("target")
        if source not in active or not isinstance(target,str) or not (target in active or target.startswith("object:")):
            raise ValueError("invalid relation owner/target")
        # Never inject a relation to an actor absent from Anima. Otherwise
        # per-person CatFlat may hallucinate a missing counterpart.
        if source not in confirmed or (target in active and target not in confirmed):
            continue
        if source==f"slot{slot}" or target==f"slot{slot}":
            relations.append(f"{source} to {target}: {rel.get('type','')} contact {rel.get('visible_geometry','')}")
    return ", ".join([
        "dupliCatFlat anime illustration; modify only the existing person; preserve original composition and exact body silhouette",
        "CANONICAL_IDENTITY_OUTFIT: "+traits,
        "EXPRESSION: "+required["expression"],
        "GAZE: "+required["gaze"],
        "ACTION: "+required["action"],
        "POSE: "+required["pose"],
        "ROLE_AND_PROP_OWNER: "+required["role"],
        "ORIGINAL_POSITION: "+required["position"],
        "CONTACT_RELATIONS: "+"; ".join(relations),
        "keep all mixed emotions, gaze, limb and object-owner geometry unchanged"
    ])


def _reject_impact_control_syntax(text):
    """Fail closed: Planner/canonical text is DATA, never Impact wildcard code.

    Reject every Impact control family, wildcard/LoRA expansions and BREAK,
    not only the three tokens that our own builder emits.
    """
    if not isinstance(text, str) or re.search(r"[\[\]{}<>]|\bBREAK\b|__|[\x00-\x1f]", text, re.IGNORECASE):
        raise ValueError("unsafe Impact wildcard directive in character prompt")


def _observed_field_known(value):
    """Unknown VLM expression/action means identity alone is not enough.

    This checks explicit uncertainty markers, not semantic correctness of the
    returned natural-language observation; real-image review remains mandatory.
    """
    if not isinstance(value,str):
        return False
    normalized=" ".join(value.strip().casefold().replace("_"," ").replace("-"," ").split())
    if normalized in ("", "?", "???", "none", "n/a", "n a", "not applicable",
                      "不明", "未確認", "判別不能", "見えない"):
        return False
    return not normalized.startswith((
        "unknown", "uncertain", "not visible", "not observed", "not observable",
        "unobservable", "occluded", "unable to determine", "cannot tell",
        "can't tell", "不明", "未確認", "判別不能", "見えない"))



# A conservative, finite vocabulary for high-risk visual cues. This is not a
# visual model: it rejects explicit text contradictions rather than proving the
# rendered pixels are faithful. Unrecognized phrases require an exact match.
_VISUAL_CUES = {
    "expression": {
        "angry": r"\b(?:angry|anger|mad|furious|scowl\w*|irritat\w*)\b",
        "embarrassed": r"\b(?:embarrass\w*|blush\w*|bashful|shy)\b",
        "surprised": r"\b(?:surpris\w*|startl\w*|shock\w*|astonish\w*)\b",
        "happy": r"\b(?:happy|happily|cheerful|delight\w*|joyful|grin\w*)\b",
        "sad": r"\b(?:sad|sorrow\w*|unhappy|tearful|cry\w*|sob\w*)\b",
        "smile": r"\b(?:smil\w*|grin\w*)\b",
        "fearful": r"\b(?:afraid|fear\w*|scared|frighten\w*)\b",
    },
    "action": {
        "jump": r"\b(?:jump\w*|leap\w*|airborne|midair|mid air)\b",
        "spin": r"\b(?:spin\w*|twirl\w*|rotat\w*)\b",
        "walk": r"\b(?:walk\w*|stroll\w*)\b",
        "run": r"\b(?:run|runs|running|sprint\w*)\b",
        "sit": r"\b(?:sit|sits|sitting|seated)\b",
        "stand": r"\b(?:stand\w*|standing|upright)\b",
        "kick": r"\b(?:kick\w*)\b",
        "throw": r"\b(?:throw\w*|toss\w*)\b",
        "catch": r"\b(?:catch\w*|caught)\b",
        "dance": r"\b(?:danc\w*)\b",
        "photograph": r"\b(?:photograph\w*|photo\w*|tak\w+ pictures?)\b",
    },
    "pose": {
        "airborne": r"\b(?:both feet off (?:the )?ground|feet in (?:the )?air|off ground|midair|mid air|airborne|leap\w*|jump\w*)\b",
        "raised_arm": r"\b(?:one arm raised|arm raised|raised arm|arm up|hand raised|raised hand|hand up)\b",
        "kneeling": r"\b(?:kneel\w*|on (?:one|both) knee\w*)\b",
        "sitting": r"\b(?:sit\w*|seated)\b",
        "upright": r"\b(?:stand\w*|upright|straight posture)\b",
    },
    "gaze": {
        "away": r"\b(?:look\w* away|gaze\w* away|looking aside|look\w* aside|avert\w* (?:gaze|eyes))\b",
        "camera": r"\b(?:toward (?:the )?camera|at (?:the )?camera|into (?:the )?camera|facing camera)\b",
        "down": r"\b(?:look\w* down|downward gaze|gaze\w* downward)\b",
        "up": r"\b(?:look\w* up|upward gaze|gaze\w* upward)\b",
    },
}
_CONTACT_OBJECTS = ("camera","phone","cup","ball","bottle","microphone",
                    "umbrella","book","bag","flower","instrument","toy",
                    "leash","staff","sword","glasses")


def _normalized_observation(value):
    return " ".join(re.sub(r"[^a-z0-9\s]", " ", value.casefold().replace("_", " ").replace("-", " ")).split())


def _visual_semantics_match(expected, observed, kind):
    """Reject unsupported/mismatched cues; cannot independently verify image."""
    if not isinstance(expected, str) or not expected.strip() or not _observed_field_known(observed):
        return False
    want,seen=_normalized_observation(expected),_normalized_observation(observed)
    # "not jumping" must not be accepted merely because it contains "jumping".
    if re.search(r"\b(?:no|not|never|without|maybe|perhaps|unclear|unsure)\b",seen):
        return False
    cues=_VISUAL_CUES[kind]
    observed_cues={key for key,pattern in cues.items() if re.search(pattern,seen)}
    # Every member of an explicitly mixed expression/action must match; a
    # recognized first cue must not hide an unsupported second cue.
    chunks=[x.strip() for x in re.split(r"\b(?:and|while|but|plus|with)\b|&|,",want) if x.strip()]
    if not chunks:
        return False
    for chunk in chunks:
        required={key for key,pattern in cues.items() if re.search(pattern,chunk)}
        if required:
            if not required.issubset(observed_cues):
                return False
        elif chunk not in seen:
            # Outside the small vocabulary: exact phrase, never invented synonym.
            return False
    # Explicit contradictions in the same VLM sentence must override isolated
    # positive cue words (e.g., "jumping but standing on the ground").
    mutually_exclusive={
        "action": ({"jump","stand"},{"jump","sit"},{"run","sit"}),
        "pose": ({"airborne","upright"},{"airborne","sitting"},
                 {"airborne","kneeling"},{"upright","sitting"}),
        "gaze": ({"away","camera"},{"up","down"})
    }
    if any(pair.issubset(observed_cues) for pair in mutually_exclusive.get(kind,())):
        return False
    if kind in ("action","pose") and (
            "jump" in observed_cues or "airborne" in observed_cues):
        if re.search(r"\b(?:feet (?:planted|on (?:the )?ground)|grounded stance)\b",seen):
            return False
    return True


def _relation_has_contact_risk(rel):
    txt=_normalized_observation(str(rel.get("type",""))+" "+
                                str(rel.get("visible_geometry",""))+" "+
                                str(rel.get("target","")))
    return (any(re.search(r"\b"+re.escape(obj)+r"\b",txt) for obj in _CONTACT_OBJECTS)
            or bool(re.search(r"\b(?:hand\w*|hold\w*|touch\w*|grasp\w*|grip\w*|hug\w*|contact|carry\w*|restrain\w*|press\w*|push\w*|pull\w*|support\w*)\b",txt)))


def _observed_contact_matches(record, rel):
    """Verify claimed owner/object tokens; this is not pixel-level protection."""
    text=record.get("observed_contact")
    if not _observed_field_known(text):
        return False
    observed=_normalized_observation(text)
    if re.search(r"\b(?:no|not|never|without|maybe|unclear|unsure)\b",observed):
        return False
    # The photographed subject is a relation target, not an owner of the
    # camera. Only visible_geometry can specify who physically holds the prop.
    declared=_normalized_observation(str(rel.get("visible_geometry","")))
    expected_owners=set(re.findall(r"\bslot[1-4]\b",declared))
    if not expected_owners:
        expected_owners={rel["source"]}
    if not all(re.search(r"\b"+re.escape(owner)+r"\b",observed) for owner in expected_owners):
        return False
    claimed_holders=set(re.findall(
        r"\b(slot[1-4])\s+(?:hold\w*|carry\w*|carrying|grasp\w*|grip\w*|use\w*)\b",
        observed))
    if claimed_holders and not claimed_holders.issubset(expected_owners):
        return False
    objects={obj for obj in _CONTACT_OBJECTS
             if re.search(r"\b"+re.escape(obj)+r"\b",declared)}
    target=rel.get("target","")
    if target.startswith("object:"):
        name=_normalized_observation(target.partition(":")[2])
        if not name or name not in observed:
            return False
    if not all(re.search(r"\b"+re.escape(obj)+r"\b",observed) for obj in objects):
        return False
    if re.search(r"\b(?:hands?|hold\w*|grasp\w*|grip\w*)\b",declared):
        if not re.search(r"\b(?:hands?|hold\w*|held|grasp\w*|grip\w*)\b",observed):
            return False
    return True


def _validate_canonical_prompt_inputs(cast, planner):
    """Validate even inputs for actors later skipped (Impact injection safety)."""
    fields=("role","action","expression","gaze","pose","position")
    for slot,tags in cast.items():
        _reject_impact_control_syntax(tags)
        for key in fields:
            # Planner may omit fields for a requested character not visible
            # after Anima; syntax-check fields that exist. Require all fields
            # later only for an actor actually approved for CatFlat.
            value=planner.get(f"slot{slot}_{key}")
            if value is None or value=="":
                continue
            if not isinstance(value,str):
                raise ValueError(f"Planner slot{slot}_{key} invalid")
            _reject_impact_control_syntax(value)
    relations=planner.get("relations",[])
    if not isinstance(relations,list):
        raise ValueError("invalid Planner relations")
    active={f"slot{s}" for s in cast}
    for rel in relations:
        if not isinstance(rel,dict):
            raise ValueError("invalid relation")
        source,target=rel.get("source"),rel.get("target")
        if source not in active or not isinstance(target,str) or not (
                target in active or target.startswith("object:")):
            raise ValueError("invalid relation owner/target")
        for value in (source,target,rel.get("type",""),rel.get("visible_geometry","")):
            _reject_impact_control_syntax(value)


def _bind_response(response,segs,cast,planner,scene,*,require_ownership=False,contact_protected=False,preserve_observed=False,character_ids=None):
    obj=_strict_json(response)
    items=obj.get("bindings")
    if not isinstance(items,list) or len(items)!=len(segs[1]):
        raise ValueError("VLM binding entries must exactly match numbered safe SEGS")
    missing=obj.get("missing_slots",[])
    if not isinstance(missing,list) or any(type(v)!=int or v not in cast for v in missing) or len(set(missing))!=len(missing):
        raise ValueError("invalid missing_slots")
    found={};used=set()
    for row in items:
        if not isinstance(row,dict):raise ValueError("invalid binding entry")
        index=row.get("instance_index")
        if type(index)!=int or not (0<=index<len(items)) or index in found:
            raise ValueError("duplicate or absent instance_index")
        found[index]=row
    if set(found)!=set(range(len(items))):raise ValueError("VLM omitted person index")
    counts=Counter(v.get("slot") for v in items if type(v.get("slot"))==int)
    # Even an UNCERTAIN detection claiming the same canonical identity is a
    # conflict: do not confidently edit only one of two potential matches.
    canonical_ids=set(character_ids.values()) if character_ids is not None else {_features(tags)["character_id"] for tags in cast.values()}
    id_counts=Counter(r.get("character_id") for r in items
                      if isinstance(r.get("character_id"),str)
                      and r.get("character_id") in canonical_ids)
    if set(missing) & set(counts):raise ValueError("missing slot contradicts visible actor")
    eligible=[];audit=[]
    for index in range(len(items)):
        r=found[index]
        slot=r.get("slot")
        evidence=r.get("identity_evidence")
        identity_ok=(type(slot)==int and slot in cast and counts[slot]==1
            and isinstance(r.get("character_id"),str)
            and id_counts[r["character_id"]]==1
            and r.get("status")=="CONFIRMED" and r.get("confidence")=="high"
            and isinstance(evidence,list) and len(evidence)>=2
            and all(isinstance(x,str) and len(x.strip())>=4 for x in evidence)
            and len({x.strip().casefold() for x in evidence})==len(evidence)
            and _matches(r,slot,cast,character_ids))
        issue=None
        if require_ownership:
            owner=r.get("region_ownership");other=r.get("other_person_pixels");proof=r.get("ownership_evidence")
            if (owner not in ("single_person","uncertain") or "other_person_pixels" not in r
                or other is not None and type(other) is not bool or not isinstance(proof,list)
                or not all(isinstance(x,str) and len(x.strip())>=4 for x in proof)
                or owner=="single_person" and (other is not False or not proof)):
                identity_ok=False
        if not identity_ok:
            issue="IDENTITY_NOT_PROVEN"
        elif not preserve_observed:
            for key,kind in (("expression","expression"),("action","action"),
                             ("pose","pose"),("gaze","gaze")):
                if not _visual_semantics_match(planner.get(f"slot{slot}_{key}"),
                                               r.get("observed_"+key),kind):
                    issue=key.upper()+"_NOT_VISUALLY_MATCHED"
                    break
        if issue is None:
            eligible.append((index,slot))
        audit.append({"instance_index":index,"slot":slot if type(slot)==int and slot in cast else None,
                      "status":"PRESENT_CONFIRMED" if issue is None else "PRESENT_UNCERTAIN",
                      "reason":issue})
    # Contact props can fall inside a person SEG. Never trust a canonical owner
    # while any relevant visible participant claims contradictory prop evidence.
    by_slot={slot:index for index,slot in eligible}
    blocked_slots=set()
    for rel in planner.get("relations",[]):
        if not _relation_has_contact_risk(rel):
            continue
        source,target=rel["source"],rel["target"]
        participants={source}
        known_actors={f"slot{s}" for s in cast}
        target_is_actor=target in known_actors
        if target_is_actor:
            participants.add(target)
        if not all(int(p[4:]) in by_slot for p in participants):
            # Merely photographing an absent person does not create shared
            # pixels. Actual touch/hand-holding does: hold back every confirmed
            # party instead of modifying a person next to an uncertain limb.
            desc=_normalized_observation(str(rel.get("type",""))+" "+
                                         str(rel.get("visible_geometry","")))
            actual_touch=(target_is_actor and (
                (re.search(r"\b"+re.escape(target)+r"\b",desc) is not None and
                 re.search(r"\b(?:hands?|arms?|wrists?|hold\w*|touch\w*|hug\w*|grasp\w*|grip\w*|restrain\w*|press\w*|push\w*|pull\w*|support\w*)\b",
                           desc) is not None)
                or re.search(r"\b(?:holding hands|handshake|hug\w*|embrac\w*|touch\w*|restrain\w*|press\w*|push\w*|pull\w*|support\w*)\b",
                             desc) is not None))
            if actual_touch and not contact_protected:
                blocked_slots.update(int(p[4:]) for p in participants
                                     if int(p[4:]) in by_slot)
            continue
        if any(not _observed_contact_matches(found[by_slot[int(p[4:])]],rel)
               for p in participants):
            blocked_slots.update(int(p[4:]) for p in participants)
    candidates=[(index,slot) for index,slot in eligible if slot not in blocked_slots]
    for slot in blocked_slots:
        entry=audit[by_slot[slot]]
        entry["status"]="PRESENT_UNCERTAIN"
        entry["reason"]="CONTACT_OWNER_OR_SHARED_PROP_UNVERIFIED"
    used_slots={slot for _,slot in candidates}
    if not candidates:raise ValueError("zero confidently bound safe people: CatFlat GPU blocked")
    # Build prompts after the *full* visible cast has been verified, so missing
    # actor relations can never be reintroduced into the prompt.
    accepted=[]
    for index,slot in candidates:
        prompt=_person_prompt(slot,cast,planner,scene,used_slots)
        _reject_impact_control_syntax(prompt)
        accepted.append((index,slot,prompt))
    # Removed uncertain SEGS cannot consume a wildcard item; exact cardinality.
    selected=(segs[0],[segs[1][i] for i,slot,prompt in accepted])
    wildcard="[ASC]"+"[SEP]".join(prompt for _,_,prompt in accepted)
    if wildcard.count("[SEP]")!=len(accepted)-1 or wildcard.endswith("[SEP]"):
        raise ValueError("unsafe Impact wildcard cardinality")
    # An unknown visible detection could be any missing logical actor. Do not
    # promote the VLM's "missing" assertion into a proven Anima disappearance.
    has_unresolved_region=any(row["status"]!="PRESENT_CONFIRMED" for row in audit)
    missing_status=[{"slot":slot,
                     "status":("MISSING_AT_ANIMA" if slot in missing and not has_unresolved_region
                               else "UNCONFIRMED"),
                     "vlm_reported_missing":slot in missing}
                    for slot in cast if slot not in used_slots]
    return selected,wildcard,{"status":"BINDINGS_VERIFIED_BY_RULES_IMAGE_NOT_YET_HUMAN_REVIEWED",
        "accepted":[{"instance_index":i,"slot":slot} for i,slot,_ in accepted],
        "people":audit,"missing":missing_status,"generated_person_count":len(accepted),
        "protection_limits":{"mask_outside_union":"COMPOSITOR_POLICY_STATIC_ONLY",
         "contact_props_inside_union":"NOT_PIXEL_PROVEN",
         "expression_action_after_catflat":"NOT_VISUALLY_VERIFIED"}}




# Stage 10: conservative background mask. No added ComfyUI node classes.
# The raw detector regions are NOT the filtered safe SEGS. Every raw person,
# including a rejected/uncertain one, must remain untouched by background edit.
_BACKGROUND_PROP_WORDS = re.compile(
    r"\b(?:cameras?|umbrellas?|frisbees?|discs?|gifts?|presents?|swords?|"
    r"weapons?|balls?|books?|phones?|smartphones?|maps?|microphones?|"
    r"bicycles?|bikes?|bottles?|glasses of water|chairs?|tables?|"
    r"puddles?|bags?|boxes?|baskets?|flowers?|drinks?|cups?|"
    r"knives?|helmets?|photographs?|props?|guitars?|instruments?|"
    r"backpacks?|handbags?|purses?|keys?|hats?|rackets?|"
    r"handheld|holding an object|carrying an object|wearing a hat)\b|"
    r"カメラ|傘|フリスビー|地図|剣|刀|ボール|プレゼント|箱|花束|花|"
    r"バッグ|かばん|鞄|スマホ|携帯|本|ギター|楽器|自転車|鍵|"
    r"椅子|テーブル|コップ|水筒|ぬいぐるみ|人形|手紙|帽子|眼鏡|"
    r"めがね|指輪|マイク|旗|弓|矢|ラケット|持ち物|小道具", re.IGNORECASE)


def _background_prop_risk(planner, scene="", observed_vlm_response=""):
    """Conservative lexical tripwire over all available sources, not proof of absence."""
    if (not isinstance(planner, dict) or not isinstance(planner.get("relations", []), list)
            or not isinstance(scene, str)
            or not isinstance(observed_vlm_response, str)):
        return True
    def gaze_text(text):
        # Looking toward the camera is a gaze direction, not proof of a
        # handheld camera in the picture. Actual camera/prop mentions elsewhere
        # remain subject to the conservative tripwire.
        return re.sub(r"\b(?:toward|towards|at|into|facing)\s+(?:the\s+)?camera\b","",text,flags=re.I)
    try:
        observation=_strict_json(observed_vlm_response)
        for row in observation.get("bindings",[]):
            if isinstance(row,dict) and isinstance(row.get("observed_gaze"),str):row["observed_gaze"]=gaze_text(row["observed_gaze"])
        risk_observation=json.dumps(observation)
    except (ValueError,TypeError):risk_observation=observed_vlm_response
    strings = [scene, risk_observation]
    for key,value in planner.items():
        if key == "relations":
            for rel in value:
                if not isinstance(rel, dict):
                    return True
                target = rel.get("target", "")
                if not isinstance(target, str):
                    return True
                if target.lower().startswith("object:"):
                    return True
                strings.extend(str(v) for v in rel.values())
        elif isinstance(value, str):
            strings.append(gaze_text(value) if key.endswith("_gaze") else value)
    return bool(_BACKGROUND_PROP_WORDS.search(" ".join(strings)))


def _background_edit_mask(raw_person_segs, confirmed_mask, planner,
                          manual_prop_mask=None, prop_review_id="", radius=48,
                          scene="", observed_vlm_response="",protect_raw_bbox=True,extra_protection=None):
    """Return one binary [1,H,W] MASK or a zero mask with explicit skip audit.

    Backend behavior is fail-closed for malformed raw regions, missing prop
    coverage, and source-ID/mask inconsistencies; person CatFlat still proceeds.
    This is NOT proof that a detector found every real figure or prop.
    """
    height,width = map(int, confirmed_mask.shape)
    blank = torch.zeros((1,height,width), dtype=torch.float32)
    audit = {
        "mode":"BACKGROUND_ONLY_MASKED_CATFLAT",
        "status":"SKIP", "reason":"NOT_VERIFIED",
        "raw_person_regions":0, "radius_px":radius,
        "all_visible_people_proven":"NO_DETECTOR_CANNOT_GUARANTEE",
        "prop_coverage_proven":"NO_MANUAL_REVIEW_REQUIRED",
        "visual_style_verified":False,
    }
    def skip(reason):
        audit["reason"] = reason
        return blank,audit
    if (raw_person_segs is None or not isinstance(raw_person_segs,(tuple,list))
            or len(raw_person_segs)!=2
            or not isinstance(raw_person_segs[0],(tuple,list))
            or tuple(raw_person_segs[0]) != (height,width)
            or not isinstance(raw_person_segs[1],(tuple,list))
            or not 1 <= len(raw_person_segs[1]) <= 12):
        return skip("RAW_PERSON_SEGS_MISSING_OR_INVALID")
    protected = torch.zeros((height,width), dtype=torch.bool)
    try:
        for seg in raw_person_segs[1]:
            # Both soft-support pixels AND entire raw bounding rectangles. Never
            # discard overlap, invalid-soft, or low-confidence person regions.
            support = _seg_mask(seg,height,width,include_soft_support=True)
            protected |= support
            x0,y0,x1,y1 = [float(x) for x in seg.bbox]
            if not all(np.isfinite(v) for v in (x0,y0,x1,y1)):
                return skip("RAW_BBOX_NONFINITE")
            lo_x,lo_y = int(np.floor(x0)),int(np.floor(y0))
            hi_x,hi_y = int(np.ceil(x1)),int(np.ceil(y1))
            if not (0 <= lo_x < hi_x <= width and 0 <= lo_y < hi_y <= height):
                return skip("RAW_BBOX_INVALID")
            if protect_raw_bbox:protected[lo_y:hi_y,lo_x:hi_x] = True
        audit["raw_person_regions"] = len(raw_person_segs[1])
    except (ValueError, TypeError, AttributeError, RuntimeError, OverflowError):
        return skip("UNSAFE_RAW_PERSON_MASK")
    if bool((confirmed_mask.bool() & ~protected).any()):
        return skip("CONFIRMED_PERSON_NOT_COVERED_BY_RAW_PROTECTION")
    if extra_protection is not None:
        extra=torch.as_tensor(extra_protection,dtype=torch.float32,device="cpu")
        if tuple(extra.shape)!=(height,width) or not torch.isfinite(extra).all() or (extra<0).any() or (extra>1).any():raise ValueError("Invalid anchored background protection")
        protected|=extra>0
    prop_risk = _background_prop_risk(
        planner, scene=scene, observed_vlm_response=observed_vlm_response)
    if prop_risk and manual_prop_mask is None:
        return skip("PROP_MASK_REQUIRED_FOR_SCENE")
    if manual_prop_mask is not None:
        if (not isinstance(prop_review_id,str)
                or len(prop_review_id.strip()) < 8):
            return skip("HUMAN_PROP_MASK_REVIEW_ID_REQUIRED")
        try:
            props = torch.as_tensor(manual_prop_mask,dtype=torch.float32,device="cpu")
            if props.ndim==3 and props.shape[0]==1:
                props=props[0]
            if props.ndim != 2 or tuple(props.shape) != (height,width):
                return skip("PROP_MASK_SIZE_MISMATCH")
            if (not bool(torch.isfinite(props).all())
                    or bool((props<0).any()) or bool((props>1).any())):
                return skip("PROP_MASK_INVALID_PIXELS")
            prop_support = props > 0
            if not bool(prop_support.any()):
                return skip("EMPTY_PROP_MASK_WITH_OBJECT_RELATIONS" if prop_risk
                            else "EMPTY_REVIEWED_PROP_MASK")
            protected |= prop_support
            audit["prop_mask_review_id"] = prop_review_id.strip()
            audit["prop_coverage_proven"] = "MANUAL_ATTESTATION_ONLY"
        except (TypeError,ValueError,RuntimeError):
            return skip("PROP_MASK_INVALID")
    # Separable max pooling avoids enormous O(radius^2) kernels on ~2MP frames.
    wpool=torch.nn.functional.max_pool2d(
        protected.float()[None,None],kernel_size=(1,2*radius+1),
        stride=1,padding=(0,radius))
    grown=torch.nn.functional.max_pool2d(
        wpool,kernel_size=(2*radius+1,1),stride=1,padding=(radius,0))[0,0]
    edit=(grown <= 0)
    audit["protected_pixels"]=int((~edit).sum().item())
    audit["background_edit_pixels"]=int(edit.sum().item())
    if not bool(edit.any()):
        return skip("NO_SAFE_BACKGROUND_PIXELS")
    audit["status"]="MASK_DEFINED_UNTESTED_ON_REAL_IMAGE"
    audit["reason"]="PERSON_AND_PROP_GUARD_APPLIED"
    return edit.float().unsqueeze(0),audit

class NexusImpactPrepare:
    @classmethod
    def INPUT_TYPES(cls):return {"required":{"image":("IMAGE",),"segs":("SEGS",)},"optional":{
        "face_segs":("SEGS",),"hand_segs":("SEGS",),"sam_model":("SAM_MODEL",),
        "mode":(["legacy","face_anchored"],{"default":"legacy"}),
        "protect_details":("BOOLEAN",{"default":True}),"manual_detail_mask":("MASK",),
        "catalogue":("STRING",{"forceInput":True}),"active_count":("INT",{"forceInput":True}),
        "identity1":("STRING",{"forceInput":True}),"identity2":("STRING",{"forceInput":True}),
        "identity3":("STRING",{"forceInput":True}),"identity4":("STRING",{"forceInput":True})}}
    RETURN_TYPES=("IMAGE","SEGS","STRING","STRING")
    RETURN_NAMES=("numbered_global_and_crops","safe_person_SEGS","rejection_report","logical_identity_ledger")
    FUNCTION="prepare"
    CATEGORY="Eagle Nexus/Existing-node bridge"

    def prepare(self,image,segs,face_segs=None,hand_segs=None,sam_model=None,mode="legacy",protect_details=True,manual_detail_mask=None,catalogue=None,active_count=None,identity1="",identity2="",identity3="",identity4=""):
        if mode=="face_anchored":
            if face_segs is None or hand_segs is None or sam_model is None:raise ValueError("Actual face/hand SEGS and installed SAM required")
            ledger=_identity_ledger(catalogue,(identity1,identity2,identity3,identity4),active_count)
            from .face_anchor import prepare_anchored
            return (*prepare_anchored(image,segs,face_segs,hand_segs,sam_model,protect_details,manual_detail_mask),ledger)
        if mode!="legacy":raise ValueError("Unknown segmentation mode")
        header, safe, rejected=_ordered_segs(segs,image)
        if not safe:raise ValueError("No safe person mask for vision identity binding")
        image_out=_overlay(image,safe)
        meta={"safe_indices":list(range(len(safe))),
              "original_detector_indices":[r["detector_index"] for r in safe],
              "skipped":rejected,"vlm_required_schema":"numbered person P0..P3"}
        return image_out,(header,[r["seg"] for r in safe]),json.dumps(meta,ensure_ascii=False),""


class NexusImpactBind:
    @classmethod
    def INPUT_TYPES(cls):return {"required":{
        "safe_segs":("SEGS",),"vlm_response":("STRING",{"forceInput":True}),
        "planner":("STRING",{"forceInput":True}),"scene":("STRING",{"forceInput":True}),
        "active_count":("INT",{"forceInput":True}),
        "logical1":("STRING",{"forceInput":True}),"logical2":("STRING",{"forceInput":True}),
        "logical3":("STRING",{"forceInput":True}),"logical4":("STRING",{"forceInput":True}),
        "raw_person_segs":("SEGS",),
        "prop_review_id":("STRING",{"default":""})},
        "optional":{"manual_prop_mask":("MASK",),"geometry_report":("STRING",{"forceInput":True}),"image":("IMAGE",),"identity_ledger":("STRING",{"forceInput":True})}}
    RETURN_TYPES=("SEGS","STRING","STRING","MASK","MASK","SEGS","STRING","MASK","MASK","BOOLEAN","BOOLEAN","BOOLEAN")
    RETURN_NAMES=("confirmed_person_SEGS","exact_wildcard","audit","verified_union_mask","background_edit_mask",
        "confirmed_face_SEGS","face_wildcard","face_union_mask","body_soft_mask","background_enabled","body_enabled","face_enabled")
    FUNCTION="bind"
    CATEGORY="Eagle Nexus/Existing-node bridge"

    def bind(self,safe_segs,vlm_response,planner,scene,active_count,logical1,logical2,logical3,logical4,raw_person_segs=None,prop_review_id="",manual_prop_mask=None,geometry_report=None,image=None,identity_ledger=None):
        if type(active_count)!=int or not 1<=active_count<=4:
            raise ValueError("expected logical people must be in 1..4")
        cast={i:v for i,v in enumerate((logical1,logical2,logical3,logical4),1) if i<=active_count}
        if any(not v.strip() for v in cast.values()):raise ValueError("empty canonical cast")
        modern=geometry_report is not None
        character_ids=_ledger_ids(identity_ledger,cast) if modern else None
        char_ids=list(character_ids.values()) if modern else [_features(value)["character_id"] for value in cast.values()]
        if len(char_ids)!=len(set(char_ids)):raise ValueError("duplicate logical identity")
        _assert_impact_asc_safe(safe_segs)
        contract=_strict_json(planner)
        _validate_canonical_prompt_inputs(cast,contract)
        height,width=map(int,safe_segs[0])
        if modern:
            from .face_anchor import capsule,raw_support,render_set,encode_mask
            import hashlib
            meta,rows,protected=capsule(geometry_report,safe_segs,image)
            actual_body=raw_support(raw_person_segs,(height,width))
            if meta.get("raw_body_hash")!=hashlib.sha256(np.packbits(actual_body).tobytes()).hexdigest():raise ValueError("Raw body changed after observation")
        selected,wildcard,audit=_bind_response(vlm_response,safe_segs,cast,contract,scene,
                                             require_ownership=modern,contact_protected=modern,preserve_observed=modern,character_ids=character_ids)
        if modern and (meta.get("skipped") or meta.get("unassigned_raw_person_regions")):
            for row in audit["missing"]:
                row["status"]="UNCONFIRMED";row["reason"]="OTHER_DETECTED_FACES_EXCLUDED_NOT_PROVEN_MISSING"
            audit["excluded_face_regions"]=meta["skipped"]
            audit["unassigned_raw_person_regions"]=meta.get("unassigned_raw_person_regions",0)
        face_set=((height,width),[]);face_wildcard="[ASC]";face_union=torch.zeros((1,height,width),dtype=torch.float32)
        if modern:
            pairs=[(x["instance_index"],x["slot"]) for x in audit["accepted"]];accepted_slots={slot for _,slot in pairs}
            observations={r["instance_index"]:r for r in _strict_json(vlm_response)["bindings"]}
            observed_plan=dict(contract);intent=[]
            for index,slot in pairs:
                observed=observations[index]
                face_known=all(_visual_semantics_match(observed.get("observed_"+kind),observed.get("observed_"+kind),kind)
                               for kind in ("expression","gaze"))
                body_known=all(_visual_semantics_match(observed.get("observed_"+kind),observed.get("observed_"+kind),kind)
                               for kind in ("action","pose"))
                body_known=body_known and observed.get("region_ownership")=="single_person" and observed.get("other_person_pixels") is False
                if not face_known:rows[index]["face"]=np.zeros((height,width),bool)
                if not body_known:rows[index]["edit"]=np.zeros((height,width),bool)
                for kind in ("expression","gaze","action","pose"):
                    value=observed.get("observed_"+kind)
                    matched=_visual_semantics_match(contract.get(f"slot{slot}_{kind}"),value,kind)
                    if not matched:intent.append({"instance_index":index,"slot":slot,"field":kind,"status":"ANIMA_INTENT_NOT_PROVEN","expected":contract.get(f"slot{slot}_{kind}"),"observed":value})
                    observed_plan[f"slot{slot}_{kind}"]=value if _observed_field_known(value) else "preserve original visible geometry"
                observed_plan[f"slot{slot}_position"]="preserve actual observed crop position; no relocation"
                observed_plan[f"slot{slot}_role"]="preserve observed role and prop ownership; do not reassign roles"
                if active_count==4:rows[index]["face"]=np.zeros((height,width),bool)
            audit["face_4P_bypass"]=active_count==4
            # Preserve observed contact; never make CatFlat repair a desired
            # dance/jump that Anima did not actually produce.
            observed_plan["relations"]=[{**rel,"type":"preserve existing observed interaction","visible_geometry":
                observations[next(i for i,s in pairs if f"slot{s}"==rel["source"])].get("observed_contact","preserve original")}
                for rel in contract.get("relations",[]) if rel.get("source") in {f"slot{s}" for s in accepted_slots}
                and (rel.get("target") not in {f"slot{s}" for s in cast} or rel["target"] in {f"slot{s}" for s in accepted_slots})]
            audit["anima_intent_mismatches"]=intent
            prop_risk=_background_prop_risk(contract,scene,vlm_response)
            valid_props=manual_prop_mask is not None and isinstance(prop_review_id,str) and bool(prop_review_id.strip())
            unresolved_contact=any(_relation_has_contact_risk(rel) and
                rel.get("target") in {f"slot{s}" for s in cast} and
                not {int(rel["source"][4:]),int(rel["target"][4:])}.issubset(accepted_slots)
                for rel in contract.get("relations",[]))
            if unresolved_contact:
                for row in rows:row["edit"]=np.zeros((height,width),bool)
                audit["body_fallback"]="CONTACT_PARTNER_UNCONFIRMED_FACE_ONLY"
            if prop_risk and not valid_props:
                for row in rows:row["edit"]=np.zeros((height,width),bool)
                audit["body_fallback"]="PROP_REGION_UNVERIFIED_FACE_ONLY"
            elif manual_prop_mask is not None:
                from .face_anchor import binary_result,dilate
                prop=binary_result(manual_prop_mask,(height,width))
                if not valid_props or not prop.any():raise ValueError("Reviewed nonempty prop mask required")
                for row in rows:
                    row["edit"]&=~dilate(prop,24);row["face"]&=~dilate(prop,8)
            selected,wildcard,body_numpy,body_map=render_set(pairs,rows,(height,width),cast,observed_plan,scene,accepted_slots,"edit")
            face_set,face_wildcard,face_numpy,face_map=render_set(pairs,rows,(height,width),cast,observed_plan,scene,accepted_slots,"face")
            for mapping in (body_map,face_map):
                for item in mapping:item["character_id"]=character_ids[item["slot"]]
            audit["logical_identity_ledger"]=_strict_json(identity_ledger)
            if not face_set[1] and not selected[1]:raise ValueError("Zero safely editable confirmed regions; CatFlat blocked")
            face_union=torch.from_numpy(face_numpy.astype(np.float32)).unsqueeze(0)
            from .safe_extent import inner_feather
            soft=torch.from_numpy(inner_feather(body_numpy,24).astype(np.float32)/255).unsqueeze(0)
            audit["render_mapping"]={"body":body_map,"face":face_map}
            audit["render_region_counts"]={"face":len(face_set[1]),"body":len(selected[1]),"confirmed_people":len(pairs)}
            audit["geometry"]={"format":meta["format"],"source_image_sha256":meta["image_sha256"],"ownership":"VLM_CLAIM_PLUS_GEOMETRY_RULES_NOT_REAL_IMAGE_PROOF",
                               "detail_protection":meta["detail_protection"],"contact":"PROTECTED_ORIGINAL_PIXELS","new_classes":0}
        verified_mask=torch.zeros((height,width),dtype=torch.float32)
        for segment in selected[1]:
            verified_mask=torch.maximum(verified_mask,_seg_mask(segment,height,width).float())
        if not bool(verified_mask.any()) and not modern:
            raise ValueError("no confirmed person pixels: mask composite must be blocked")
        background_support=torch.maximum(verified_mask,face_union[0])
        use_anchored_background=(modern and not meta.get("skipped") and not meta.get("body_rejected")
            and not meta.get("unassigned_raw_person_regions") and len(pairs)==len(rows)
            and all(observations[i].get("region_ownership")=="single_person" and observations[i].get("other_person_pixels") is False for i,_ in pairs)
            and meta.get("person_extent")!="FACE_ONLY_NO_BODY_DETECTION")
        background_mask,background_report=_background_edit_mask(
            raw_person_segs,background_support,contract,
            manual_prop_mask=manual_prop_mask,prop_review_id=prop_review_id,
            scene=scene,observed_vlm_response=vlm_response,protect_raw_bbox=not use_anchored_background,
            extra_protection=protected if modern else None)
        audit["background"] = background_report
        if modern:
            audit["background"]["protection_policy"]="RAW_ALL_PERSON_SUPPORT_PLUS_ANCHORED_GUARDS_DILATED" if use_anchored_background else "RAW_BBOX_CONSERVATIVE_FALLBACK"
            if not use_anchored_background:
                background_mask.zero_();audit["background"]["status"]="SKIP";audit["background"]["disabled_reason"]="UNRESOLVED_FACE_BODY_COVERAGE"
        if not modern:soft=verified_mask.unsqueeze(0)
        return (selected,wildcard,json.dumps(audit,ensure_ascii=False,indent=2),
                verified_mask.unsqueeze(0),background_mask,face_set,face_wildcard,face_union,soft,
                bool(background_mask.any()),bool(verified_mask.any()),bool(face_union.any()))


NODE_CLASS_MAPPINGS={"NexusImpactPrepare":NexusImpactPrepare,"NexusImpactBind":NexusImpactBind}
NODE_DISPLAY_NAME_MAPPINGS={"NexusImpactPrepare":"Nexus | 全身SEGS番号付き観測","NexusImpactBind":"Nexus | 本人SEGS確定とCatFlat分離"}
