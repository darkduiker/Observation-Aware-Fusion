"""Patient-weighted fusion controls; all fitting uses development patients."""
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit
from sklearn.model_selection import StratifiedKFold


def patient_weights(ids):
    _, inv, counts = np.unique(ids, return_inverse=True, return_counts=True)
    return 1.0 / counts[inv]


def loss(y, p, weights, kind='logloss'):
    p = np.clip(np.asarray(p), 1e-6, 1-1e-6)
    values = (y-p)**2 if kind == 'brier' else -(y*np.log(p)+(1-y)*np.log1p(-p))
    return float(np.average(values, weights=weights))


class Fusion:
    def __init__(self, mode, strength=0.0):
        self.mode, self.strength = mode, float(strength)

    def fit(self, current, history, observation, reliability, target, ids):
        w = patient_weights(ids)
        self.mean = np.mean(observation, axis=0)
        self.std = np.std(observation, axis=0)
        self.std[self.std < 1e-6] = 1.0
        if self.mode == 'fixed_half':
            self.alpha = 0.5
            return self
        if self.mode == 'global':
            r = minimize_scalar(lambda a: loss(target, current+a*(history-current), w),
                                bounds=(0,1), method='bounded')
            self.alpha = float(r.x)
            return self
        if self.mode == 'original':
            x = np.column_stack([np.ones(len(observation)), np.clip(observation[:,:4], -3,3)])
        else:
            x = np.column_stack([np.ones(len(observation)), np.clip((observation-self.mean)/self.std,-3,3)])
        reliable = reliability if self.mode == 'reliable_shrink' else np.ones(len(current))
        def objective(beta):
            g = reliable * expit(x@beta)
            p = current+g*(history-current)
            reg = 0.02*np.sum(beta[1:]**2)+0.01*beta[0]**2
            shrink = self.strength*np.average(g*g, weights=w)
            return loss(target,p,w)+reg+shrink
        fit = minimize(objective,np.zeros(x.shape[1]),method='L-BFGS-B')
        if not fit.success and not np.isfinite(fit.fun):
            raise RuntimeError(fit.message)
        self.beta = fit.x
        self.optimizer_success = bool(fit.success)
        return self

    def gates(self, obs, reliability):
        if self.mode in ['fixed_half','global']:
            return np.full(len(obs),self.alpha)
        if self.mode == 'original':
            x = np.column_stack([np.ones(len(obs)),np.clip(obs[:,:4],-3,3)])
        else:
            x = np.column_stack([np.ones(len(obs)),np.clip((obs-self.mean)/self.std,-3,3)])
        r = reliability if self.mode == 'reliable_shrink' else np.ones(len(obs))
        return r*expit(x@self.beta)

    def predict(self, c, h, obs, reliability):
        return c+self.gates(obs,reliability)*(h-c)


def select_gate(mode,c,h,obs,r,y,ids,seed):
    """Select shrinkage on meta folds inside the outer training pool only."""
    unique = np.unique(ids)
    labels = np.array([y[np.flatnonzero(ids==i)[0]] for i in unique])
    rows=[]
    for strength in [0.0,0.01,0.05,0.2,1.0]:
        pred=np.zeros(len(y))
        for a,b in StratifiedKFold(3,shuffle=True,random_state=seed).split(unique,labels):
            tr=np.isin(ids,unique[a]); va=np.isin(ids,unique[b])
            gate=Fusion(mode,strength).fit(c[tr],h[tr],obs[tr],r[tr],y[tr],ids[tr])
            pred[va]=gate.predict(c[va],h[va],obs[va],r[va])
        rows.append({'strength':strength,'meta_brier':loss(y,pred,patient_weights(ids),'brier')})
    selected=min(rows,key=lambda x:(x['meta_brier'],-x['strength']))['strength']
    return Fusion(mode,selected).fit(c,h,obs,r,y,ids),rows
