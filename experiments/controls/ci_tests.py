"""Real edit (untrained rows to zero) against (a) the 20 freq-matched placebo edits and (b) no change.
Full test set, 7,608 samples, paired bootstrap of the samples, 10,000 resamples, margins recentred as in splits.py."""
import json, numpy as np
B=10000; models=[0,2,3,4,5,8,12,13]
def f1(tp,pp,ap): return 100*2*tp/np.maximum(pp+ap,1)
out={}
for no in models:
    z=np.load('out/logits_M%d.npz'%no); y=z['y'].astype(bool); n=len(y)
    mu=z['orig'].mean()
    rc=lambda a: a+(mu-a.mean())
    P={'orig':z['orig']>0,'uz':rc(z['untrained_zero'])>0}
    for k in range(20): P['p%d'%k]=rc(z['placebo_freqmatch_%d'%k])>0
    rng=np.random.RandomState(20261117); acc={k:[] for k in P}
    for _ in range(B//250):
        idx=rng.randint(0,n,size=(250,n)); apos=y[idx].sum(1)
        for k,p in P.items(): acc[k].append(f1((p&y)[idx].sum(1),p[idx].sum(1),apos))
    F={k:np.concatenate(v) for k,v in acc.items()}
    pe=lambda p: float(f1((p&y).sum(),p.sum(),y.sum()))
    pt={k:pe(p) for k,p in P.items()}
    eff=F['uz']-F['orig']; plac=np.mean([F['p%d'%k] for k in range(20)],0)-F['orig']; dif=eff-plac
    def s(v):
        lo,hi=np.percentile(v,[2.5,97.5]); p=2*min((v<=0).mean(),(v>=0).mean()); return {'ci95':[float(lo),float(hi)],'p':float(max(p,1/(B+1)))}
    pl_pt=[pt['p%d'%k]-pt['orig'] for k in range(20)]
    out[no]={'real_point':pt['uz']-pt['orig'],'real_vs_none':s(eff),'placebo_mean_point':float(np.mean(pl_pt)),'placebo_min_max':[float(min(pl_pt)),float(max(pl_pt))],
             'real_minus_placebo':s(dif),'n_placebo_ge_real':int(sum(v>=pt['uz']-pt['orig'] for v in pl_pt)),'placebo_vs_none':s(plac)}
    print(no,json.dumps(out[no]))
json.dump(out,open('out/ci_tests.json','w'),indent=1)
