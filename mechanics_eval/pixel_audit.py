"""Independent audit-only RGB patch translation; no semantic labels or actions.

Excludes the entire tested table column and player. Fixed pixel gates are deliberately
strict and this check may abstain when terrain is corrupted. It does not rescore.
"""
import numpy as np

def align(a,b):
 out=[]
 for dr in range(-2,3):
  for dc in range(-2,3):
   support=[];bad=[];patterns=[]
   for r in range(7):
    for c in range(9):
     rr,cc=r-dr,c-dc
     if not(0<=rr<7 and 0<=cc<9) or c == 5 or cc == 5 or (r,c) == (3,4) or (rr,cc) == (3,4):continue
     p=a[r*7:r*7+7,c*7:c*7+7].astype(float);q=b[rr*7:rr*7+7,cc*7:cc*7+7].astype(float)
     # Pixel texture, rather than inferred semantic tile classes.
     if max(p.std((0,1)).mean(),q.std((0,1)).mean()) < 12:continue
     d=np.sqrt(np.mean((p-q)**2))
     if d<=3:
      support.append([r,c,rr,cc,round(float(d),3)])
      if not any(np.sqrt(((p-x)**2).mean())<10 for x in patterns):patterns.append(p)
     else:bad.append([r,c,rr,cc,round(float(d),3)])
   n=len(support)+len(bad);out.append(dict(shift=[dr,dc],support=support,conflicts=bad,types=len(patterns),agreement=len(support)/n if n else 0))
 out.sort(key=lambda x:(x['agreement'],len(x['support'])),reverse=True)
 good=[x for x in out if len(x['support'])>=2 and x['types']>=2]
 best=good[0] if good else out[0];margin=best['agreement']-(good[1]['agreement'] if len(good)>1 else 0)
 return dict(accepted=bool(good and best['agreement']>=.8 and margin>=.2),margin=margin,**best)
