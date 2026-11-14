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


def sample_frames(path, n=60):
    """n cadre distribuite uniform pe durata videoclipului"""
    c=cv2.VideoCapture(path); N=int(c.get(7)); fr=[]
    for i in np.linspace(0,N-1,n).astype(int):
        c.set(1,int(i)); ok,f=c.read()
        if ok: fr.append(f)
    c.release()
    if not fr: raise RuntimeError('videoclipul nu a putut fi citit')
    return fr

def background(path, n=60, frames=None):
    fr=frames if frames is not None else sample_frames(path,n)
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

TOL = 0.045          # eroarea acceptata la potrivire, ca fractiune din perioada marcajului (0,8 m la 18 m)
DUTY_BIAS_RAYS = 0.035   # liniutele par mai lungi din cauza neclaritatii imaginii (masurat pe clipuri cu marcaj cunoscut)

def find_dashed_lines(bg):
    """Liniile discontinue ale unui drum drept: fiecare este o raza din punctul de fuga, cu capetele liniutelor
    periodice in u = 1/d. Intoarce (punct de fuga, lista de linii); lista este goala daca nu se gaseste nimic."""
    g,th,mask=marking_maps(bg); bl=blobs_from_mask(mask)
    try:
        vp,inl=vanishing_point(bl); rs=rays(bl,inl,vp)
    except Exception:
        return None,[]
    found=[]
    for r in rs:
        d,v=profile(th,vp,r['angle'],g.shape); e=dash_edges(d,v)
        if e is None or e['quality']<0.8 or e['n']<6 or not 0.12<e['duty']<0.7 or not 12<r['angle']<168: continue
        dirv=np.array([np.cos(np.radians(r['angle'])),np.sin(np.radians(r['angle']))])
        u=np.r_[e['ua'],e['ub']]
        found.append(dict(angle=r['angle'],pts=vp[None]+dirv[None]/u[:,None],duty=e['duty'],edges=e))
    return vp,found

def measured_duty(found):
    """raportul liniuta / (liniuta + pauza), masurat pe marcaj (nu depinde de scara)"""
    return float(np.median([r['duty'] for r in found]))-DUTY_BIAS_RAYS

def fit_lines(found, shape, vp, period, dash, lane):
    """Potriveste modelul camerei pe liniile gasite, pentru dimensiunile date ale marcajului."""
    if len(found)<2: raise RuntimeError('calibrare automata nereusita: nu s-au gasit cel putin doua linii discontinue')
    tol=TOL*period
    found=[dict(r) for r in found]
    for r in found:
        e=r['edges']; nom=np.r_[e['ka']*period, e['kb']*period+dash]; r['nom']=nom-nom.min()
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
        cam=fit_camera(ray_pts,shape,vp,period,dash,lane)
        errs=[]
        for i,(gi,li,pts,nom) in enumerate(ray_pts):
            w=_apply(cam['H'],pts); gidx=cam['sides'].index(gi)
            errs.append(float(np.sqrt(np.mean((w[:,0]-(cam['X0'][gidx]+lane*li))**2+(w[:,1]-(cam['phi'][i]+nom))**2))))
        worst=int(np.argmax(errs))
        if (errs[worst]<=max(tol,3*np.median(errs)) and cam['rmse']<=tol) or len(ray_pts)<=2: break
        bad=ray_pts[worst]; found=[r for r in found if not (r['side']==bad[0] and r['lane']==bad[1])]
        ray_pts=[]
        for gi in sorted({r['side'] for r in found}):
            for li,r in enumerate([r for r in found if r['side']==gi]): r['lane']=li; ray_pts.append((gi,li,r['pts'],r['nom']))
    cam['ray_err']=errs
    if cam['rmse']>tol: raise RuntimeError('calibrare automata nereusita: eroare %.2f m'%cam['rmse'])
    return dict(vp=vp,rays=found,cam=cam)

def calibrate(bg, period=18.0, dash=6.0, lane=3.75):
    vp,found=find_dashed_lines(bg)
    return fit_lines(found,bg.shape[:2],vp,period,dash,lane)


# ---------------- scara reala, estimata din vehicule ----------------
# Marcajele dau geometria drumului, dar nu si dimensiunile lor reale (acestea difera de la o tara la alta).
# O estimare independenta a scarii vine din vehicule: latimea unei masini este cunoscuta, iar pe benzile laterale
# caseta ei include si flancul, cu atat mai mult cu cat masina este mai aproape de camera.

CAR_WIDTH, CAR_FLANK = 1.95, 3.4     # latimea aparenta a casetei unei masini vazute din fata / lungimea vizibila a flancului (m)

def _car_extent(H, cars):
    """latimea casetei la nivelul solului si pozitia masinii (X lateral, Y in lungul drumului, camera in origine)"""
    bl=_apply(H,cars[:,[0,3]]); br=_apply(H,cars[:,[2,3]]); c=(bl+br)/2
    return np.abs(br[:,0]-bl[:,0]),c[:,0],np.abs(c[:,1])

def scale_from_cars(H, cars):
    """Drum cu mai multe benzi: scara laterala este cunoscuta (latimea benzii), cea longitudinala nu.
    Intoarce factorul s cu care trebuie inmultite distantele in lungul drumului, sau None daca masinile nu ajung."""
    if cars is None or len(cars)<40: return None
    W,X,Y=_car_extent(H,cars); ok=(W>1.2)&(W<4.5)
    ok&=Y<=np.median(Y[ok])              # doar jumatatea apropiata: acolo flancul se vede cel mai bine
    W,X,Y=W[ok],X[ok],Y[ok]
    if len(W)<20 or np.ptp(X)<5: return None
    def res(q):
        s,w0,x0=q
        return W-(w0+np.maximum(np.abs(X-x0)-w0/2,0)*CAR_FLANK/(s*Y+CAR_FLANK))
    r=least_squares(res,[1.0,2.0,0.0],bounds=([0.05,1.2,-30],[20,3.0,30]),loss='soft_l1',f_scale=0.15)
    return float(r.x[0])


# ---------------- drum cu o singura linie discontinua (eventual in curba) ----------------

EDGE_BLUR_PX = 2.0      # cu atat apare mai lunga o liniuta in masca, din cauza neclaritatii marginilor

def dash_blobs(mask):
    """petele albe alungite si pline (liniute de marcaj), cu capatul indepartat si cel apropiat"""
    n,lab,st,_=cv2.connectedComponentsWithStats(mask,8); out=[]
    for i in range(1,n):
        if st[i,4]<40: continue
        x,y,w,h=st[i,:4]; ys,xs=np.where(lab[y:y+h,x:x+w]==i); pts=np.float64(np.c_[xs+x,ys+y]); m=pts.mean(0)
        _,_,vt=np.linalg.svd(pts-m,full_matrices=False)
        t=(pts-m)@vt[0]; wv=(pts-m)@vt[1]; L=t.max()-t.min(); W=wv.max()-wv.min()+1
        if L/W>=2.5 and len(pts)/(L*W)>0.6 and L>=12:
            a=vt[0] if vt[0][1]>0 else -vt[0]          # axa orientata in jos (spre camera)
            out.append(dict(m=m,a=a,L=L,W=W,far=m-a*L/2,near=m+a*L/2))
    return out

def dash_chains(bl):
    """inlantuie liniutele asezate cap la cap (aceeasi directie, marimi apropiate); cele mai lungi lanturi primele"""
    nxt={}
    for i,b in enumerate(bl):
        best=None
        for j,c in enumerate(bl):
            d=c['m']-b['m']; al=d@b['a']
            if i==j or al<=0: continue
            if abs(d[0]*b['a'][1]-d[1]*b['a'][0])>0.2*al+3 or abs(b['a']@c['a'])<np.cos(np.radians(15)): continue
            if not 0.5<c['L']/b['L']<2.6 or not 0.5<c['W']/b['W']<2.6: continue
            if al-(b['L']+c['L'])/2<0.3*min(b['L'],c['L']) or al>7*max(b['L'],c['L']): continue
            if best is None or al<best[0]: best=(al,j)
        if best: nxt[i]=best[1]
    prev={}
    for i,j in nxt.items():       # fiecare liniuta are un singur predecesor: cel mai apropiat
        if j not in prev or np.linalg.norm(bl[i]['m']-bl[j]['m'])<np.linalg.norm(bl[prev[j]]['m']-bl[j]['m']): prev[j]=i
    nxt={i:j for j,i in prev.items()}; out=[]
    for s in range(len(bl)):
        if s in prev or s not in nxt: continue
        ch=[s]
        while ch[-1] in nxt: ch.append(nxt[ch[-1]])
        out.append(ch)
    return sorted(out,key=lambda ch:(len(ch),sum(bl[i]['L'] for i in ch)),reverse=True)

def straight_dash_line(mask, iters=4000, tol=6.0, seed=0):
    """Liniute scurte (in imagine aproape patrate, cand camera priveste razant) asezate pe o dreapta, la distante care
    variaza lent. Intoarce (capete indepartate, capete apropiate) pentru cel mai lung sir regulat, sau None."""
    n,lab,st,cen=cv2.connectedComponentsWithStats(mask,8)
    ids=[i for i in range(1,n) if 6<=st[i,4]<=6000 and st[i,4]>=0.5*st[i,2]*st[i,3]]
    if len(ids)<5: return None
    c=cen[ids]; area=st[ids,4].astype(float); rng=np.random.default_rng(seed); best=(0,None)
    big=np.where(area>=20)[0]
    if len(big)<5: return None
    for _ in range(iters):
        i,j=rng.choice(big,2,replace=False); d=c[j]-c[i]; L=np.hypot(*d)
        if L<30: continue
        d=d/L; inl=np.abs((c[big,0]-c[i,0])*d[1]-(c[big,1]-c[i,1])*d[0])<tol; score=np.sqrt(area[big][inl]).sum()
        if score>best[0]: best=(score,big[inl])
    if best[1] is None or len(best[1])<5: return None
    m=c[best[1]].mean(0); _,_,vt=np.linalg.svd(c[best[1]]-m,full_matrices=False); d=vt[0] if vt[0][1]>0 else -vt[0]
    inl=np.where(np.abs((c[:,0]-m[0])*d[1]-(c[:,1]-m[1])*d[0])<tol)[0]
    inl=inl[np.argsort(-((c[inl]-m)@d))]               # de la aproape spre departe: liniutele se micsoreaza treptat
    # petele mult mai mici decat liniuta precedenta (butoni reflectorizanti, zgomot) sunt sarite
    start=max(inl[:max(len(inl)//4,3)],key=lambda k:area[k]); kept=[start]
    for k in inl[list(inl).index(start)+1:]:
        if 0.45<area[k]/area[kept[-1]]<1.3: kept.append(k)
    # cel mai lung sir in care pasul dintre liniute se schimba lent (o liniuta lipsa dubleaza pasul: este acceptata)
    runs=[[kept[0]]]
    for k in kept[1:]:
        r=runs[-1]; step=(c[r[-1]]-c[k])@d; ok=step>1.2*np.sqrt(area[k])*0.5
        if ok and len(r)>=2: ok=0.4<step/((c[r[-2]]-c[r[-1]])@d)<1.25
        if ok: r.append(k)
        else: runs.append([k])
    run=max(runs,key=len)[::-1]                        # inapoi in ordinea departe -> aproape
    if len(run)<5: return None
    far,near=[],[]
    for k in run:
        i=ids[k]; x,y,w,h=st[i,:4]; ys,xs=np.where(lab[y:y+h,x:x+w]==i); t=(np.c_[xs+x,ys+y]-c[k])@d
        far.append(c[k]+d*t.min()); near.append(c[k]+d*(t.max()+1))
    return np.array(far),np.array(near)

def find_single_lines(bg):
    """liniile discontinue candidate, cea cu cele mai multe liniute prima: [(capete indepartate, capete apropiate)]"""
    mask=marking_maps(bg)[2]; out=[]
    bl=dash_blobs(mask); chains=dash_chains(bl)
    if chains and len(chains[0])>=4:
        out.append((np.array([bl[i]['far'] for i in chains[0]]),np.array([bl[i]['near'] for i in chains[0]])))
    line=straight_dash_line(mask)
    if line is not None: out.append(line)
    return sorted(out,key=lambda l:len(l[0]),reverse=True)

def _H_tilt(f, tilt, h, cx, cy):
    """omografia imagine -> drum pentru o camera fara rotire in jurul axei optice, inclinata cu 'tilt' sub orizontala"""
    s,c=np.sin(tilt),np.cos(tilt)
    return np.array([[h,0,-h*cx],[0,-h*s,h*(f*c+cy*s)],[0,c,f*s-cy*c]])

def fit_single_line(far, near, shape, f):
    """Potriveste inclinarea si inaltimea camerei (pentru focala f) astfel incat liniutele sa fie egale si echidistante.
    Unitatea de lungime a rezultatului este perioada marcajului (liniuta + pauza)."""
    hh,ww=shape; cx,cy=ww/2,hh/2
    def world(q):
        H=_H_tilt(f,q[0],np.exp(q[1]),cx,cy); return _apply(H,near),_apply(H,far)
    def equal_lengths(q):
        a,b=world(q); L=np.hypot(*(a-b).T); return L/np.median(L)-1
    best=None
    for t0 in np.radians([10,25,45,65]):
        r=least_squares(equal_lengths,[t0,0.0],bounds=([0.02,-5],[1.55,5]))
        if best is None or r.cost<best.cost: best=r
    a,b=world(best.x); L=np.median(np.hypot(*(a-b).T))
    steps=np.r_[np.hypot(*np.diff(a,axis=0).T),np.hypot(*np.diff(b,axis=0).T)]/L
    base=np.median(steps[steps<1.6*steps.min()])
    n=np.maximum(np.round((steps[:len(a)-1]+steps[len(a)-1:])/2/base),1)       # liniute lipsa -> pas de 2, 3... perioade
    def periodic(q):
        a,b=world(q); L=np.hypot(*(a-b).T)
        return np.r_[np.hypot(*np.diff(a,axis=0).T)-n,np.hypot(*np.diff(b,axis=0).T)-n,0.5*(L-q[2])]
    q0=np.r_[best.x,1/base]; q0[1]-=np.log(L*base)
    lo,hi=np.array([0.02,-9,0.05]),np.array([1.55,9,0.9]); q0=np.clip(q0,lo+1e-6,hi-1e-6)
    r=least_squares(periodic,q0,bounds=(lo,hi),loss='soft_l1',f_scale=0.1)
    H=_H_tilt(f,r.x[0],np.exp(r.x[1]),cx,cy)
    a=_apply(H,near); d=a[-1]-a[0]; ang=np.arctan2(d[0],d[1]); c,s=np.cos(ang),np.sin(ang)
    H=np.array([[c,-s,0],[s,c,0],[0,0,1.]])@H                                  # axa Y = directia liniei
    # raportul liniuta/perioada: din liniutele cele mai mari din imagine (cel mai putin afectate de neclaritate)
    a,b=world(r.x); Lw=np.hypot(*(a-b).T); Lpx=np.hypot(*(near-far).T); top=np.argsort(Lpx)[-max(len(Lpx)//3,3):]
    duty=float(np.median(Lw[top]*(Lpx[top]-EDGE_BLUR_PX)/Lpx[top]))
    return dict(H=H,f=float(f),h=float(np.exp(r.x[1])),duty=duty,
                rmse=float(np.sqrt(np.mean(periodic(r.x)[:2*len(n)]**2))))

def cars_near_line(cars, far, near, k=1.5):
    """masinile de pe benzile alaturate liniei (la cel mult k latimi de masina de ea)"""
    if cars is None or not len(cars): return np.zeros((0,4))
    m=(near+far)/2; L=np.hypot(*(near-far).T); o=np.argsort(m[:,1]); m,L=m[o],L[o]
    bx,by,bw=(cars[:,0]+cars[:,2])/2,cars[:,3],cars[:,2]-cars[:,0]
    ok=(np.abs(bx-np.interp(by,m[:,1],m[:,0]))<k*bw)&(by>=m[0,1]-L[0])&(by<=m[-1,1]+L[-1])
    return cars[ok]

def period_from_cars(H, cars):
    """O singura linie: toata scara este necunoscuta. Intoarce (perioada in metri pentru care casetele corespund unor
    masini reale, abaterea potrivirii)."""
    W,X,Y=_car_extent(H,cars)
    def res(q):
        P=np.exp(q[0])
        return np.log(W*P/(CAR_WIDTH+np.maximum(np.abs(X-q[1])*P-CAR_WIDTH/2,0)*CAR_FLANK/(Y*P+CAR_FLANK)))
    best=None
    for p0 in (5,10,20):
        r=least_squares(res,[np.log(p0),0.0],loss='soft_l1',f_scale=0.08)
        if best is None or r.cost<best.cost: best=r
    return float(np.exp(best.x[0])),best.cost/len(W)

def calibrate_single_line(bg, cars=None):
    """Calibrare pe o singura linie discontinua. Focala nu poate fi dedusa dintr-o singura linie: se alege cea pentru
    care casetele masinilor se potrivesc cel mai bine (sau o valoare uzuala, daca nu sunt destule masini).
    Intoarce dict cu H (in perioade), duty, rmse (in perioade), period_est (metri, din masini; None daca lipsesc)."""
    lines=find_single_lines(bg); shape=bg.shape[:2]
    if not lines: raise RuntimeError('calibrare automata nereusita: nu s-a gasit nicio linie discontinua')
    for far,near in lines:
        sel=cars_near_line(cars,far,near)
        if len(sel)<8:
            fit=fit_single_line(far,near,shape,1.2*shape[1]); fit['period_est']=None
        else:
            tries=[]
            for f in shape[1]*np.geomspace(0.5,8.0,13):
                fit=fit_single_line(far,near,shape,f); P,cost=period_from_cars(fit['H'],sel); tries.append((cost,fit,P))
            k=int(np.argmin([t[0] for t in tries])); fit=tries[k][1]
            # optim la marginea intervalului = masinile nu determina focala (filmare cu teleobiectiv): scara ramane necunoscuta
            fit['period_est']=tries[k][2] if 0<k<len(tries)-1 else None
        if fit['rmse']<=0.15:
            fit['pts']=np.vstack([far,near]); return fit
    raise RuntimeError('calibrare automata nereusita: linia discontinua gasita nu este regulata')
