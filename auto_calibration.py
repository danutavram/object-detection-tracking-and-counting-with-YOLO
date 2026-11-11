# auto_calibration.py
# Calibrarea automata a camerei pe baza marcajelor rutiere discontinue.
#
# Etape:
#   1. background()      - imaginea drumului fara vehicule (mediana cadrelor, cu compensarea miscarii camerei);
#   2. marking_maps()    - evidentierea marcajelor albe (transformarea top-hat);
#   3. vanishing_point() - punctul de fuga al directiei drumului (RANSAC pe segmentele de marcaj);
#   4. rays()/profile()  - fiecare linie de marcaj este o raza ce porneste din punctul de fuga;
#   5. dash_edges()      - pe o raza, in variabila u = 1/d (d = distanta fata de punctul de fuga), liniutele
#                          unui marcaj discontinuu sunt periodice; se determina capetele fiecarei liniute;
#   6. fit_camera()      - se potriveste un model de camera (focala, rotatie, inaltime) astfel incat capetele
#                          liniutelor sa ajunga la pozitiile lor reale (lungime liniuta, perioada, latime banda);
#   7. calibrate()       - leaga etapele si elimina razele care nu sunt marcaje de banda (parapet, rosturi).
#
# Rezultatul este omografia imagine -> planul drumului (metri).
import cv2
import numpy as np
from scipy.ndimage import maximum_filter1d, minimum_filter1d
from scipy.optimize import least_squares


def background(path, n=60):
    c=cv2.VideoCapture(path); N=int(c.get(7)); fr=[]
    for i in np.linspace(0,N-1,n).astype(int):
        c.set(1,int(i)); ok,f=c.read()
        if ok: fr.append(f)
    c.release()
    if not fr: raise RuntimeError('videoclipul nu a putut fi citit')
    # compensarea micilor deplasari ale camerei: puncte caracteristice urmarite (Lucas-Kanade) + RANSAC,
    # astfel incat vehiculele in miscare sa nu influenteze estimarea
    ref=cv2.cvtColor(fr[0],cv2.COLOR_BGR2GRAY)
    p0=cv2.goodFeaturesToTrack(ref,800,0.01,15)
    sh=np.array([global_shift(ref,p0,cv2.cvtColor(f,cv2.COLOR_BGR2GRAY)) for f in fr])
    sh-=np.median(sh,0)          # pozitia de referinta = pozitia dominanta a camerei pe durata clipului
    out=[cv2.warpAffine(f,np.float32([[1,0,-dx],[0,1,-dy]]),(f.shape[1],f.shape[0]),borderMode=cv2.BORDER_REPLICATE) for f,(dx,dy) in zip(fr,sh)]
    background.drift=float(np.hypot(*sh.T).max())
    return np.median(np.stack(out),0).astype(np.uint8)

def global_shift(ref_gray, p0, gray):
    """deplasarea globala (dx, dy) a cadrului fata de referinta, robusta la vehiculele in miscare"""
    if p0 is None: return (0.0,0.0)
    p1,st,_=cv2.calcOpticalFlowPyrLK(ref_gray,gray,p0,None,winSize=(31,31),maxLevel=3)
    ok=st.ravel()==1
    if ok.sum()<20: return (0.0,0.0)
    d=(p1-p0)[ok].reshape(-1,2); med=np.median(d,0); inl=np.hypot(*(d-med).T)<1.5
    if inl.sum()<0.3*len(d): return (0.0,0.0)
    return tuple(float(v) for v in d[inl].mean(0))

def marking_maps(bg):
    g=cv2.cvtColor(bg,cv2.COLOR_BGR2GRAY)
    th=cv2.morphologyEx(g,cv2.MORPH_TOPHAT,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(41,41)))
    sat=cv2.cvtColor(bg,cv2.COLOR_BGR2HSV)[...,1]
    mask=((th>35)&(sat<60)&(g>140)).astype(np.uint8)*255
    return g,th,mask

def blobs_from_mask(mask):
    n,lab,st,_=cv2.connectedComponentsWithStats(mask,8); out=[]
    for i in range(1,n):
        if st[i,4]<25: continue
        ys,xs=np.where(lab==i); pts=np.float64(np.c_[xs,ys]); m=pts.mean(0)
        _,_,vt=np.linalg.svd(pts-m,full_matrices=False)
        t=(pts-m)@vt[0]; w=(pts-m)@vt[1]; L=t.max()-t.min(); W=4*w.std()+1
        if L/W>=2.5: out.append((m+vt[0]*t.min(), m+vt[0]*t.max(), L))
    return out

def vanishing_point(blobs, iters=3000, tol=0.012, seed=0):
    rng=np.random.default_rng(seed)
    lines=[]; 
    for p1,p2,L in blobs:
        l=np.cross([*p1,1],[*p2,1]); l/=np.hypot(l[0],l[1]); lines.append((l,(p1+p2)/2,L))
    best=(0,None)
    def score(v):
        s=0; inl=[]
        for k,(l,m,L) in enumerate(lines):
            d=v-m; dn=np.hypot(*d)
            if dn<1: continue
            # unghiul dintre segment si directia spre v
            ang=abs(l[0]*v[0]+l[1]*v[1]+l[2])/dn
            if ang<tol: s+=L; inl.append(k)
        return s,inl
    for _ in range(iters):
        i,j=rng.choice(len(lines),2,replace=False,p=np.array([l[2] for l in lines])/sum(l[2] for l in lines))
        v=np.cross(lines[i][0],lines[j][0])
        if abs(v[2])<1e-9: continue
        v=v[:2]/v[2]; s,inl=score(v)
        if s>best[0]: best=(s,v,inl)
    _,v,inl=best
    # rafinare prin cele mai mici patrate ponderate
    A=np.array([lines[k][0][:2]*lines[k][2] for k in inl]); b=-np.array([lines[k][0][2]*lines[k][2] for k in inl])
    v=np.linalg.lstsq(A,b,rcond=None)[0]
    s,inl=score(v)
    return v,inl

def rays(blobs, inl, vp, gap_deg=0.8):
    """grupeaza segmentele dupa unghiul fata de punctul de fuga -> linii de marcaj"""
    items=[]
    for k,(p1,p2,L) in enumerate(blobs):
        m=(p1+p2)/2
        if m[1]<vp[1]+15: continue
        dv=m-vp; db=p2-p1
        mis=abs(np.degrees(np.arcsin(np.clip(abs(dv[0]*db[1]-dv[1]*db[0])/(np.hypot(*dv)*np.hypot(*db)),0,1))))
        if mis>6: continue
        items.append((np.degrees(np.arctan2(m[1]-vp[1],m[0]-vp[0])),L,k))
    items.sort(); groups=[[items[0]]]
    for it in items[1:]:
        if it[0]-groups[-1][-1][0]<gap_deg: groups[-1].append(it)
        else: groups.append([it])
    out=[]
    for g in groups:
        a=np.average([i[0] for i in g],weights=[i[1] for i in g])
        out.append(dict(angle=a,total=sum(i[1] for i in g),n=len(g),ids=[i[2] for i in g]))
    return out

def profile(th, vp, angle, shape, d0=60, step=0.5):
    """profilul de luminozitate (top-hat) de-a lungul razei; latimea benzii creste cu distanta"""
    h,w=shape; dirv=np.array([np.cos(np.radians(angle)),np.sin(np.radians(angle))]); nrm=np.array([-dirv[1],dirv[0]])
    ds=[];vals=[]; d=d0
    while True:
        p=vp+dirv*d
        if not(1<=p[0]<w-2 and 1<=p[1]<h-2):
            if not ds and d<4000: d+=step; continue
            break
        hw=max(1.5,0.012*d)   # semilatime banda
        offs=np.arange(-hw,hw+0.01,0.5)
        q=p[None]+offs[:,None]*nrm[None]
        xi=np.clip(q[:,0].round().astype(int),0,w-1); yi=np.clip(q[:,1].round().astype(int),0,h-1)
        vals.append(th[yi,xi].max()); ds.append(d); d+=step
    return np.array(ds),np.array(vals,float)

def _lattice_fit(ua, ub, p):
    def lattice(x,p):
        ph=np.angle(np.mean(np.exp(2j*np.pi*x/p)))/(2*np.pi)*p
        return np.round((x-ph)/p),ph
    a=0
    for _ in range(3):
        ka,_=lattice(ua,p); A=np.c_[ka,np.ones_like(ka)]
        if len(set(ka))<2: return None
        (p,a),*_=np.linalg.lstsq(A,ua,rcond=None)
    ka=np.round((ua-a)/p); kb=np.round((ub-a)/p-0.5)
    delta=np.mean(ub-a-kb*p); ra=ua-(a+ka*p); rb=ub-(a+kb*p+delta)
    ok_a=np.abs(ra)<0.12*p; ok_b=np.abs(rb)<0.12*p
    if ok_a.sum()<2 or ok_b.sum()<2: return None
    return dict(p=p,a=a,delta=delta,duty=delta/p,quality=(ok_a.mean()+ok_b.mean())/2,n=int(ok_a.sum()+ok_b.sum()),
                ua=ua[ok_a],ka=ka[ok_a],ub=ub[ok_b],kb=kb[ok_b],rms=float(np.sqrt(np.mean(np.r_[ra[ok_a],rb[ok_b]]**2))/p))

def dash_edges(d, v, dmin=220):
    """Detecteaza capetele liniutelor pe o raza. In u=1/d (d = distanta fata de punctul de fuga) capetele sunt periodice."""
    m=d>=dmin; d=d[m]; v=v[m]
    if len(d)<200: return None
    u=1/d; ug=np.linspace(u.min(),u.max(),4000); vg=np.interp(ug,u[::-1],v[::-1]); du=ug[1]-ug[0]
    def edges(thr):
        b=(vg>thr).astype(int); e=np.diff(b)
        def sub(idx):
            t=thr[idx] if np.ndim(thr) else thr
            return ug[idx]+float(np.clip((t-vg[idx])/(vg[idx+1]-vg[idx]+1e-9),0,1))*du
        r=np.where(e==1)[0]; f=np.where(e==-1)[0]
        # perechi (inceput, sfarsit) complete; se elimina petele scurte (in u toate liniutele au aceeasi lungime)
        runs=[(i,f[f>i][0]) for i in r if (f>i).any()]
        if not runs: return np.array([]),np.array([])
        L=np.array([j-i for i,j in runs]); ref=np.median(np.sort(L)[len(L)//2:])
        runs=[(i,j) for (i,j),l in zip(runs,L) if 0.5*ref<=l<=1.8*ref]
        return np.array([sub(i) for i,_ in runs]),np.array([sub(j) for _,j in runs])
    best=None
    # varianta 1: perioada din autocorelatie, prag local
    x=vg-vg.mean(); ac=np.correlate(x,x,'full')[len(x)-1:]; ac/=ac[0]+1e-9
    lag=next((k for k in range(20,len(ac)//2) if ac[k]>0.3 and ac[k]==ac[max(k-15,0):k+16].max()),None)
    if lag is not None:
        win=int(1.2*lag)|1; hi=maximum_filter1d(vg,win); lo=minimum_filter1d(vg,win)
        if np.median(hi-lo)>=25:
            ua,ub=edges((hi+lo)/2)
            if len(ua)>=2 and len(ub)>=2: best=_lattice_fit(ua,ub,lag*du)
    # varianta 2 (putine liniute): prag global, perioada din distanta dintre inceputurile liniutelor
    if best is None or best['quality']<0.8:
        lo,hi=np.percentile(vg,10),np.percentile(vg,99)
        if hi-lo>=40:
            ua,ub=edges((lo+hi)/2)
            ua=ua[(ua>ug[5])&(ua<ug[-5])]; ub=ub[(ub>ug[5])&(ub<ug[-5])]
            if 3<=len(ua)<=40 and len(ub)>=3:
                alt=_lattice_fit(ua,ub,np.median(np.diff(ua)))
                if alt and (best is None or alt['quality']>best['quality']): best=alt
    return best

# ---------------- modelul camerei si potrivirea pe marcaje ----------------

def _H_img2road(f, rvec, h, cx, cy):
    """omografia imagine -> plan drum (metri) pentru o camera cu focala f, rotatie rvec, inaltime h"""
    K=np.array([[f,0,cx],[0,f,cy],[0,0,1.]]); R,_=cv2.Rodrigues(np.asarray(rvec,float))
    C=np.array([0,0,h.item() if hasattr(h,'item') else h])
    P=K@R@np.c_[np.eye(3)[:,:2],-C]      # drum (X,Y,1) -> imagine
    return np.linalg.inv(P)

def _apply(H, pts):
    q=np.c_[pts,np.ones(len(pts))]@H.T; return q[:,:2]/q[:,2:3]

def fit_camera(ray_pts, shape, vp, period=18.0, dash=6.0, lane=3.75):
    """ray_pts: lista de (side, lane_index, pts_img (n,2), nominal_Y (n,)) pentru fiecare linie discontinua.
    Parametri: f, rvec, h + pentru fiecare sens X0 si pentru fiecare raza faza phi."""
    hgt,wid=shape; cx,cy=wid/2,hgt/2
    sides=sorted({r[0] for r in ray_pts}); nr=len(ray_pts)
    def unpack(p):
        return p[0],p[1:4],p[4],p[5:5+len(sides)],p[5+len(sides):]
    def resid(p):
        f,rv,h,X0,phi=unpack(p); H=_H_img2road(f,rv,h,cx,cy); out=[]
        for i,(s,li,pts,nomY) in enumerate(ray_pts):
            w=_apply(H,pts)
            out.append(w[:,0]-(X0[sides.index(s)]+lane*li)); out.append(w[:,1]-(phi[i]+nomY))
        return np.concatenate(out)
    best=None
    for f0 in (0.5*wid,0.8*wid,1.3*wid,2.0*wid):
        Yc=np.array([(vp[0]-cx)/f0,(vp[1]-cy)/f0,1.0]); Yc/=np.linalg.norm(Yc)
        Xc=np.cross(Yc,np.array([0,-1.,0])); Xc/=np.linalg.norm(Xc); Zc=np.cross(Xc,Yc)
        rv0,_=cv2.Rodrigues(np.c_[Xc,Yc,Zc])
        for h0 in (5.,10.):
            p0=np.r_[f0,rv0.ravel(),h0,np.zeros(len(sides)),np.zeros(nr)]
            # initializare faze / X0 din proiectia initiala
            H0=_H_img2road(f0,rv0.ravel(),h0,cx,cy)
            for i,(s_,li,pts,nomY) in enumerate(ray_pts):
                w=_apply(H0,pts); p0[5+len(sides)+i]=np.median(w[:,1]-nomY)
                if li==0: p0[5+sides.index(s_)]=np.median(w[:,0])
            lo=np.full(len(p0),-np.inf); hi=np.full(len(p0),np.inf); lo[0],hi[0]=0.3*wid,6*wid; lo[4],hi[4]=1.5,80
            try: r=least_squares(resid,p0,bounds=(lo,hi),loss='huber',f_scale=1.0,max_nfev=3000,x_scale='jac')
            except Exception: continue
            if best is None or r.cost<best.cost: best=r
    f,rv,h,X0,phi=unpack(best.x); res=resid(best.x)
    return dict(f=f,rvec=rv,h=h,X0=X0,phi=phi,H=_H_img2road(f,rv,h,cx,cy),rmse=float(np.sqrt(np.mean(res**2))*np.sqrt(2)),sides=sides)

def calibrate(bg, period=18.0, dash=6.0, lane=3.75):
    g,th,mask=marking_maps(bg); bl=blobs_from_mask(mask); vp,inl=vanishing_point(bl); rs=rays(bl,inl,vp)
    found=[]
    for r in rs:
        d,v=profile(th,vp,r['angle'],g.shape); e=dash_edges(d,v)
        if e is None or e['quality']<0.8 or e['n']<6 or not 0.12<e['duty']<0.7 or not 12<r['angle']<168: continue
        dirv=np.array([np.cos(np.radians(r['angle'])),np.sin(np.radians(r['angle']))])
        u=np.r_[e['ua'],e['ub']]; nom=np.r_[e['ka']*period, e['kb']*period+dash]
        pts=vp[None]+dirv[None]/u[:,None]
        found.append(dict(angle=r['angle'],pts=pts,nom=nom-nom.min(),duty=e['duty'],edges=e))
    if len(found)<2: raise RuntimeError('calibrare automata nereusita: nu s-au gasit cel putin doua linii discontinue')
    # carosabile: razele se impart la cel mai mare salt de unghi, daca este mult mai mare decat celelalte
    found.sort(key=lambda r:r['angle'], reverse=True)
    gaps=[found[i]['angle']-found[i+1]['angle'] for i in range(len(found)-1)]
    groups=[found]
    if len(found)>=3:
        j=int(np.argmax(gaps)); rest=[g for i,g in enumerate(gaps) if i!=j]
        if gaps[j]>2.5*np.median(rest): groups=[found[:j+1],found[j+1:]]
    ray_pts=[]
    for gi,sel in enumerate(groups):
        for li,r in enumerate(sel): r['side']=gi; r['lane']=li; ray_pts.append((gi,li,r['pts'],r['nom']))
    # potrivire iterativa: razele care nu sunt marcaje de banda (parapet, rosturi) au erori mari si sunt eliminate
    while True:
        cam=fit_camera(ray_pts,g.shape,vp,period,dash,lane)
        errs=[]
        for i,(gi,li,pts,nom) in enumerate(ray_pts):
            w=_apply(cam['H'],pts); gidx=cam['sides'].index(gi)
            errs.append(float(np.sqrt(np.mean((w[:,0]-(cam['X0'][gidx]+lane*li))**2+(w[:,1]-(cam['phi'][i]+nom))**2))))
        worst=int(np.argmax(errs))
        if (errs[worst]<=max(0.8,3*np.median(errs)) and cam['rmse']<=0.8) or len(ray_pts)<=2: break
        bad=ray_pts[worst]; found=[r for r in found if not (r['side']==bad[0] and r['lane']==bad[1])]
        ray_pts=[]
        for gi in sorted({r['side'] for r in found}):
            for li,r in enumerate([r for r in found if r['side']==gi]): r['lane']=li; ray_pts.append((gi,li,r['pts'],r['nom']))
    cam['ray_err']=errs
    if cam['rmse']>0.8: raise RuntimeError('calibrare automata nereusita: eroare %.2f m'%cam['rmse'])
    return dict(vp=vp,rays=found,cam=cam)
