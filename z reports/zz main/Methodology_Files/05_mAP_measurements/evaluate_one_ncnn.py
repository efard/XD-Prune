#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json,math,shutil,time,traceback
from pathlib import Path
from typing import Any
import yaml
from ultralytics import YOLO

GEN_NAMES=["articulated_truck","bicycle","bus","car","motorcycle","motorized_vehicle","non_motorized_vehicle","pedestrian","pickup_truck","single_unit_truck","work_van"]
SNOW_NAMES=["person","rider","car","truck","bus","train","motorcycle","bicycle"]

def norm(v:Any):
    if isinstance(v,dict):
        def k(i):
            try:return (0,int(i[0]))
            except:return (1,str(i[0]))
        return [str(x) for _,x in sorted(v.items(),key=k)]
    if isinstance(v,list): return [str(x) for x in v]
    return []

def mv(m,p):
    v=m
    for c in p.split("."):
        v=getattr(v,c,None)
        if v is None:return None
    try:
        f=float(v); return f if math.isfinite(f) else None
    except:return None

def vec(v):
    if v is None:return []
    if hasattr(v,"tolist"):v=v.tolist()
    if not isinstance(v,list):return []
    out=[]
    for x in v:
        try:out.append(float(x))
        except:out.append(float("nan"))
    return out

def imgsz(md):
    v=md.get("imgsz",640)
    if isinstance(v,(list,tuple)):
        if len(v)!=2 or int(v[0])!=int(v[1]): raise RuntimeError(f"Unsupported imgsz {v}")
        return int(v[0])
    return int(v)

def alias(model_dir,root,jid):
    root.mkdir(parents=True,exist_ok=True); a=root/f"{jid}_ncnn_model"
    if a.is_symlink() or a.exists():
        if a.is_symlink() or a.is_file(): a.unlink()
        else: shutil.rmtree(a)
    a.symlink_to(model_dir.resolve(),target_is_directory=True); return a

def main():
    p=argparse.ArgumentParser()
    for name in ["job-id","relative-name"]: p.add_argument("--"+name,required=True)
    p.add_argument("--model-dir",type=Path,required=True)
    p.add_argument("--gen-data",type=Path,required=True)
    p.add_argument("--snow-data",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    a=p.parse_args()
    out=a.output_dir; out.mkdir(parents=True,exist_ok=True)
    rf=out/f"{a.job_id}.json"; cf=out/f"{a.job_id}_classes.csv"

    try:
        md=yaml.safe_load((a.model_dir/"metadata.yaml").read_text(encoding="utf-8"))
        names=norm(md.get("names")); size=imgsz(md)

        if names==GEN_NAMES: domain="GEN"; data=a.gen_data
        elif names==SNOW_NAMES: domain="SNOW"; data=a.snow_data
        else:
            started=time.time(); YOLO(str(alias(a.model_dir,out/"_aliases",a.job_id)),task="detect")
            rf.write_text(json.dumps({"job_id":a.job_id,"model":a.relative_name,"domain":"COCO80","imgsz":size,"classes":len(names),
                                      "status":"runtime_ok_no_project_map","error":"Runtime load passed; COCO-80 has no compatible project mAP.",
                                      "wall_time_seconds":time.time()-started},indent=2),encoding="utf-8")
            return
        
        started=time.time(); model=YOLO(str(alias(a.model_dir,out/"_aliases",a.job_id)),task="detect")
        metrics=model.val(data=str(data.resolve()),split="val",imgsz=size,batch=1,device="cpu",workers=0,
                          half=False,rect=True,conf=0.001,iou=0.70,max_det=300,augment=False,plots=False,
                          save_json=False,verbose=True,project=str(out/"ultralytics_runs"),name=a.job_id,exist_ok=True)

        speed=getattr(metrics,"speed",{}) or {}
        row={"job_id":a.job_id,"model":a.relative_name,"domain":domain,"imgsz":size,"classes":len(names),"status":"ok","error":"",
             "map50_95":mv(metrics,"box.map"),"map50":mv(metrics,"box.map50"),"map75":mv(metrics,"box.map75"),
             "precision":mv(metrics,"box.mp"),"recall":mv(metrics,"box.mr"),
             "preprocess_ms_per_image":speed.get("preprocess"),"inference_ms_per_image":speed.get("inference"),
             "postprocess_ms_per_image":speed.get("postprocess"),"wall_time_seconds":time.time()-started,"dataset_yaml":str(data.resolve())}

        rf.write_text(json.dumps(row,indent=2),encoding="utf-8")
        box=getattr(metrics,"box",None); maps=vec(getattr(box,"maps",None)); pv=vec(getattr(box,"p",None)); rv=vec(getattr(box,"r",None))

        with cf.open("w",newline="",encoding="utf-8") as h:
            w=csv.DictWriter(h,fieldnames=["job_id","model","domain","imgsz","class_id","class_name","map50_95","precision","recall"]); w.writeheader()
            for i,n in enumerate(names):
                w.writerow({"job_id":a.job_id,"model":a.relative_name,"domain":domain,"imgsz":size,"class_id":i,"class_name":n,
                            "map50_95":maps[i] if i<len(maps) else "","precision":pv[i] if i<len(pv) else "","recall":rv[i] if i<len(rv) else ""})

    except Exception as e:
        traceback.print_exc()
        rf.write_text(json.dumps({"job_id":a.job_id,"model":a.relative_name,"domain":"","imgsz":"","classes":"","status":"failed","error":repr(e)},indent=2),encoding="utf-8")
        raise

if __name__=="__main__": main()
