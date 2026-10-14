"""Configurable, fail-closed actor-identity / garment policy for Eagle Nexus.

A VLM may call a tie an 'accessory' even when the canonical catalog lists
it with the outfit. That is NOT evidence that a different character was
detected. Hair, eye color and face adornments remain strictly checked.

Changing presets or user-supplied *wardrobe-only* aliases via ComfyUI inputs
does not require modifying Python. Actual changes to Python still require
ComfyUI to reload the package.
"""
from __future__ import annotations

import json
import re


MODES=("balanced","strict","face_only")
_HAIR_STYLES={
    "bangs","ahoge","ponytail","twintails","braid","braided_hair",
    "hair_bun","single_side_bun","side_ponytail","hime_cut",
    "hair_over_one_eye","hair_over_eyes",
}
_EYE_NONCOLORS={"closed_eyes","half_closed_eyes","wide_eyes","narrow_eyes","half_open_eyes"}
_UNKNOWN={"","unknown","none","not_visible","uncertain","n/a","not_observed"}
_ALIAS_NECKWEAR=("tie","necktie","neck_tie","collar","scarf","ribbon")
_TOKEN=re.compile(r"^[a-z][a-z0-9_]{0,79}$")


def _token(value):
    if not isinstance(value,str):
        return None
    value=value.strip().casefold().replace("-","_").replace(" ","_")
    return value if _TOKEN.fullmatch(value) else None


def _neckwear(t):
    if t=="necktie" or t=="neck_tie":
        return "tie"
    for suffix in ("_necktie","_neck_tie"):
        if t.endswith(suffix):
            return t[:-len(suffix)]+"_tie"
    return t


def _is_face_accessory(t):
    return (any(fragment in t for fragment in
                ("hairclip","hairpin","barrette","hair_ornament","hair_ribbon","hair_bow",
                 "headband","hairband","eyeglasses","sunglasses"))
            or t in ("glasses","earrings","eyepatch","monocle","facemask"))


def _is_aliasable_wear(t):
    if _is_face_accessory(t) or t.endswith(("_hair","_eyes")):
        return False
    return t in _ALIAS_NECKWEAR or any(t.endswith("_"+x) for x in _ALIAS_NECKWEAR)


def _no_duplicate_pairs(pairs):
    result={}
    for key,value in pairs:
        if key in result:
            raise ValueError("Duplicate policy key")
        result[key]=value
    return result


def parse_policy(payload=None, *, modern=True):
    """Validate a small JSON policy. Never accept aliasing of face identity."""
    default={"mode":"balanced" if modern else "strict","wardrobe_aliases":{}}
    if payload is None or payload=="":
        return default
    if not isinstance(payload,str) or len(payload)>10000:
        raise ValueError("Invalid Nexus attribute policy JSON")
    try:
        obj=json.loads(payload,object_pairs_hook=_no_duplicate_pairs)
    except (TypeError,ValueError) as e:
        raise ValueError("Invalid Nexus attribute policy JSON") from e
    if not isinstance(obj,dict) or set(obj)-{"mode","wardrobe_aliases"}:
        raise ValueError("Unsupported Nexus attribute policy keys")
    mode=obj.get("mode",default["mode"])
    if mode not in MODES:
        raise ValueError("Unknown Nexus attribute policy mode")
    aliases=obj.get("wardrobe_aliases",{})
    if not isinstance(aliases,dict) or len(aliases)>40:
        raise ValueError("Invalid wardrobe aliases")
    mapped={}
    for original,replacement in aliases.items():
        a,b=_token(original),_token(replacement)
        if a is None or b is None or not (_is_aliasable_wear(a) and _is_aliasable_wear(b)):
            raise ValueError("Wardrobe aliases can remap neckwear only")
        # User-defined aliases must never silently change a garment's color.
        # e.g. teal_neck_ribbon -> teal_tie is a naming choice, whereas
        # teal_tie -> red_tie is an identity/outfit conflict.
        pa,pb=_neckwear(a),_neckwear(b)
        prefix_a=pa.rsplit("_",1)[0] if "_" in pa else ""
        prefix_b=pb.rsplit("_",1)[0] if "_" in pb else ""
        if prefix_a!=prefix_b:
            raise ValueError("Wardrobe aliases must preserve descriptive prefix")
        mapped[pa]=pb
    if not modern and mode!="strict":
        raise ValueError("Legacy Nexus does not support relaxed identity policy")
    return {"mode":mode,"wardrobe_aliases":mapped}


def _profile(cast_tags):
    # Canonical model tags can contain parentheses, weights or author-specific
    # syntax; ignore unparseable *non-identity* tokens rather than treating
    # them as visual garment contradictions. Never derive ID from a tag order.
    raw=[x.strip().casefold() for x in cast_tags.split(",")]
    if len(raw)<2 or not raw[1]:
        raise ValueError("Invalid canonical actor tokens")
    tokens=[_token(x) for x in raw]
    traits={x for x in tokens[2:] if x is not None}
    hair={x for x in traits if x.endswith("_hair") or x in _HAIR_STYLES}
    eyes={x for x in traits if x.endswith("_eyes")}
    eye_colors=eyes-_EYE_NONCOLORS
    if "heterochromia" in traits or len(eye_colors)>1:
        eyes.add("heterochromia")
    face={x for x in traits if _is_face_accessory(x)}
    clothes={_neckwear(x) for x in traits-hair-eyes-face if x!="heterochromia"}
    return {"id":tokens[1] or raw[1],"hair":hair,"eyes":eyes,"face":face,"clothes":clothes,
            "eye_colors":eye_colors}


def _observed_features(record):
    observed=record.get("observed_features")
    if not isinstance(observed,dict):
        return None,"FEATURE_OBJECT_MISSING"
    arrays={}
    for key in ("hair","eyes","accessory","outfit"):
        values=observed.get(key,[])
        if not isinstance(values,list) or len(values)>40 or any(not isinstance(v,str) for v in values):
            return None,"INVALID_"+key.upper()+"_FEATURES"
        normalized=set()
        for value in values:
            if value.strip().casefold() in ("n/a","not applicable"):
                continue
            t=_token(value)
            if t is None:
                return None,"INVALID_"+key.upper()+"_FEATURE"
            if t not in _UNKNOWN:
                normalized.add(t)
        arrays[key]=normalized
    all_access=arrays["accessory"]|arrays["outfit"]
    seen={
        "hair":arrays["hair"],
        "eyes":arrays["eyes"],
        "face":{x for x in all_access if _is_face_accessory(x)},
        "clothes":{_neckwear(x) for x in all_access if not _is_face_accessory(x)},
    }
    return seen,None


def verify_person_features(record,slot,cast,character_ids=None,policy=None):
    """Return identity decision and a separate body-edit permission.

    This is NOT visual proof. It validates the existing VLM's observation and
    preserves the independently segmented face when only apparel is unclear.
    """
    cfg=policy or parse_policy()
    if type(slot)!=int or slot not in cast:
        return {"ok":False,"reason":"INVALID_LOGICAL_SLOT","body_allowed":False,"unverified_wardrobe":[]}
    profiles={i:_profile(tags) for i,tags in cast.items()}
    expected=profiles[slot]
    expected_id=(character_ids or {}).get(slot,expected["id"])
    if record.get("character_id")!=expected_id:
        return {"ok":False,"reason":"CHARACTER_ID_MISMATCH","body_allowed":False,"unverified_wardrobe":[]}
    if record.get("ambiguous") is not False:
        return {"ok":False,"reason":"AMBIGUOUS_IDENTITY","body_allowed":False,"unverified_wardrobe":[]}

    seen,error=_observed_features(record)
    if error:
        return {"ok":False,"reason":error,"body_allowed":False,"unverified_wardrobe":[]}
    if not seen["hair"].issubset(expected["hair"]):
        return {"ok":False,"reason":"HAIR_CONTRADICTED","body_allowed":False,"unverified_wardrobe":[]}
    if not seen["eyes"].issubset(expected["eyes"]):
        return {"ok":False,"reason":"EYE_COLOR_CONTRADICTED","body_allowed":False,"unverified_wardrobe":[]}
    if not seen["face"].issubset(expected["face"]):
        return {"ok":False,"reason":"FACE_ACCESSORY_CONTRADICTED","body_allowed":False,"unverified_wardrobe":[]}
    if len(expected["eye_colors"])>1:
        actual_colors={x for x in seen["eyes"] if x.endswith("_eyes") and x not in _EYE_NONCOLORS}
        if actual_colors!=expected["eye_colors"]:
            return {"ok":False,"reason":"HETEROCHROMIA_COLORS_INCOMPLETE","body_allowed":False,"unverified_wardrobe":[]}

    aliases=cfg["wardrobe_aliases"]
    seen_clothes={aliases.get(x,x) for x in seen["clothes"]}
    unverified=sorted(seen_clothes-expected["clothes"])
    groups=sum(bool(seen[x]) for x in ("hair","eyes","face"))+bool(seen_clothes)
    if groups<2:
        return {"ok":False,"reason":"TOO_FEW_VISIBLE_FEATURE_CATEGORIES","body_allowed":False,"unverified_wardrobe":[]}

    def candidates(include_clothes):
        return [i for i,p in profiles.items()
                if seen["hair"].issubset(p["hair"])
                and seen["eyes"].issubset(p["eyes"])
                and seen["face"].issubset(p["face"])
                and (not include_clothes or seen_clothes.issubset(p["clothes"]))]

    if unverified:
        if cfg["mode"]=="strict":
            return {"ok":False,"reason":"UNSUPPORTED_WARDROBE:"+",".join(unverified)[:128],
                    "body_allowed":False,"unverified_wardrobe":unverified}
        # A self-reported actor ID and outfit do not suffice as face evidence.
        # Two independently visible classes (hair+eyes) must uniquely resolve.
        if not seen["hair"] or not seen["eyes"] or candidates(False)!=[slot]:
            return {"ok":False,"reason":"FACE_IDENTITY_NOT_UNIQUE_WITH_WARDROBE_MISMATCH",
                    "body_allowed":False,"unverified_wardrobe":unverified}
        return {"ok":True,"reason":"FACE_MATCH_WARDROBE_UNVERIFIED",
                "body_allowed":False,"unverified_wardrobe":unverified}
    if candidates(True)!=[slot]:
        return {"ok":False,"reason":"NOT_UNIQUE_BY_VISIBLE_FEATURES",
                "body_allowed":False,"unverified_wardrobe":[]}
    if cfg["mode"]=="face_only":
        return {"ok":True,"reason":"FACE_ONLY_POLICY",
                "body_allowed":False,"unverified_wardrobe":[]}
    return {"ok":True,"reason":"MATCHED","body_allowed":True,
            "unverified_wardrobe":[]}


class NexusAttributePolicy:
    """Selectable safeguards for the existing NexusImpactBind optional port."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required":{
            "mode":(["balanced","strict","face_only"],{"default":"balanced"}),
            "wardrobe_aliases_json":("STRING",{"default":"{}","multiline":True}),
        }}

    RETURN_TYPES=("STRING",)
    RETURN_NAMES=("attribute_policy",)
    FUNCTION="build"
    CATEGORY="Eagle Nexus/Existing-node bridge"

    def build(self,mode="balanced",wardrobe_aliases_json="{}"):
        try:
            aliases=json.loads(wardrobe_aliases_json,object_pairs_hook=_no_duplicate_pairs)
        except (TypeError,ValueError) as e:
            raise ValueError("Invalid wardrobe aliases JSON") from e
        policy=parse_policy(json.dumps({"mode":mode,"wardrobe_aliases":aliases}))
        return (json.dumps(policy,sort_keys=True),)
