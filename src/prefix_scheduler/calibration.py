"""Fit a small nonnegative lookup predictor from paired prefill measurements."""
import math
from statistics import median

class LookupPredictor:
    def __init__(self, points):
        if not points: raise ValueError('empty calibration data')
        self.points = {(int(p['length']), int(p['cached'])): float(p['saving_ms']) for p in points}
        for (l,c), v in self.points.items():
            if l <= 0 or c < 0 or c > l or not math.isfinite(v): raise ValueError('invalid calibration point')
        self.lengths = sorted(set(l for l,c in self.points))
        self.fractions = sorted(set(c/l for l,c in self.points))
    def predict(self, length, cached):
        if not isinstance(length,int) or not isinstance(cached,int) or length <= 0 or cached < 0 or cached > length: return None
        if cached == 0: return 0.0
        f=cached/length
        if not (self.lengths[0] <= length <= self.lengths[-1] and self.fractions[0] <= f <= self.fractions[-1]): return None
        # Inverse-distance interpolation over (length, fraction), preserving exact points.
        if (length,cached) in self.points: return max(0.0,self.points[(length,cached)])
        scale=max(self.lengths[-1]-self.lengths[0],1)
        neighbors=sorted(((math.hypot((l-length)/scale,c/l-f),v) for (l,c),v in self.points.items()))[:4]
        if neighbors[0][0] == 0: return max(0.0, neighbors[0][1])
        weights=[1/(d*d) for d,_ in neighbors]
        return max(0.0,sum(w*v for w,(_,v) in zip(weights,neighbors))/sum(weights))
    def to_dict(self):
        return {'method':'inverse_distance_length_and_fraction_v1','points':[{'length':l,'cached':c,'saving_ms':v} for (l,c),v in sorted(self.points.items())]}
    @classmethod
    def from_dict(cls, obj):
        if obj['method']!='inverse_distance_length_and_fraction_v1': raise ValueError('unknown predictor')
        return cls(obj['points'])

def fit(pairs):
    grouped={}
    for p in pairs:
        l,c=int(p['length']),int(p['cached'])
        cold,warm=float(p['cold_prefill_ms']),float(p['warm_prefill_ms'])
        if not all(math.isfinite(x) and x>=0 for x in (cold,warm)) or l<=0 or c<=0 or c>=l: raise ValueError('invalid timing pair')
        if int(p['warm_cached_tokens']) != c: raise ValueError('observed cache match differs from requested length')
        grouped.setdefault((l,c),[]).append(cold-warm)
    return LookupPredictor([{'length':l,'cached':c,'saving_ms':median(values)} for (l,c),values in grouped.items()])

def validate(predictor, pairs):
    results=[]
    for p in pairs:
        predicted=predictor.predict(int(p['length']),int(p['cached']))
        actual=float(p['cold_prefill_ms'])-float(p['warm_prefill_ms'])
        results.append({'length':p['length'],'cached':p['cached'],'predicted_ms':predicted,'observed_ms':actual,'error_ms':None if predicted is None else predicted-actual})
    errors=[abs(x['error_ms']) for x in results if x['error_ms'] is not None]
    comparable = mistakes = 0
    covered = [r for r in results if r['predicted_ms'] is not None]
    for i, left in enumerate(covered):
        for right in covered[i+1:]:
            observed = left['observed_ms'] - right['observed_ms']
            predicted = left['predicted_ms'] - right['predicted_ms']
            if observed == 0: continue
            comparable += 1
            if observed * predicted <= 0: mistakes += 1
    return {'count':len(results),'covered':len(errors),'mae_ms':sum(errors)/len(errors) if errors else None,
            'ranking_comparisons':comparable,'ranking_mistakes':mistakes,'predictions':results}
