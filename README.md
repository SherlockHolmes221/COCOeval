# Details about coco evaluation api
### The usage for CocoEval is as follows:
```
cocoGt=..., cocoDt=...       # load dataset and results
E = CocoEval(cocoGt,cocoDt); # initialize CocoEval object
E.params.recThrs = ...       # set parameters as desired
E.evaluate()                 # run per image evaluation
E.accumulate()               # accumulate per image results
E.summarize()                # display summary metrics of results
 ```
 
### default paras

| name | default | choices | explanation |
| ------ | ------ | ------ | ------ |
| imgIds | - | - | [all] N img ids to use for evaluation |
| catIds | - | - | [all] K cat ids to use for evaluation |
| iouThrs | [.5, 0.95] | - | thresholds for evaluation |
| recThrs | [0.0, 1.00] | - | R=101 recall thresholds for evaluation|
| maxDets | [1, 10, 100] | - | M=3 thresholds on max detections per image|
| areaRng | [[0 ** 2, 1e5 ** 2], [0 ** 2, 32 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]| - | - |
| areaRngLbl | ['all', 'small', 'medium', 'large'] | - | A=4 object area ranges for evaluation|
| useCats | 1 | - | [1] if true use category labels for evaluation |
| iouType | - | 'segm', 'bbox' or 'keypoints' | |

### Evaluate
- evaluate(self)

evaluates detections on every image and every category and concats the results into the "evalImgs" with fields:
```
# get gt and dt info
gts = self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
dts = self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
for gt in gts:
    self._gts[gt['image_id'], gt['category_id']].append(gt)
for dt in dts:
    self._dts[dt['image_id'], dt['category_id']].append(dt)
    
# computeIoU for each the size of self.ious[i] is (len(d), len(g)) 
self.ious = {(imgId, catId): computeIoU(imgId, catId) \
             for imgId in p.imgIds for catId in catIds}
```
- evaluateImg(self, imgId, catId, aRng, maxDet)
```
get gt and dt
set ignore labels and sort gt
sort the dt according to scores
for each p.iouThrs:
    for each dt:
        get the gt index as m according to ious
            dtIg[tind, dind] = gtIg[m]
            dtm[tind, dind] = gt[m]['id']
            gtm[tind, m] = d['id']
```
- result:
```
dtIds      - [1xD] id for each of the D detections (dt)
gtIds      - [1xG] id for each of the G ground truths (gt)
dtMatches  - [TxD] matching gt id at each IoU or 0
gtMatches  - [TxG] matching dt id at each IoU or 0
dtScores   - [1xD] confidence of each dt
gtIgnore   - [1xG] ignore flag for each gt
dtIgnore   - [TxD] ignore flag for each dt at each IoU
```
- accumulate(self, p=None)
```
for each cats:
    for each area:
        for each maxDet:
            sort all scores in all images using det
            concat dtm and get the tps fps
            get tp_sum and fp_sum
            for each iouThrs:
                rc = tp / npig
                pr = tp / (fp + tp + np.spacing(1))
                recall[t, k, a, m] = rc[-1]
                inds = np.searchsorted(rc, p.recThrs, side='left')
                for ri, pi in enumerate(inds):
                    q[ri] = pr[pi]
                    ss[ri] = dtScoresSorted[pi]
                precision[t, :, k, a, m] = np.array(q)
                scores[t, :, k, a, m] = np.array(ss)
```
- result:
```
params     - parameters used for evaluation
date       - date evaluation was performed
counts     - [T,R,K,A,M] parameter dimensions (see above)
precision  - [TxRxKxAxM] precision for every evaluation setting
recall     - [TxKxAxM] max recall for every evaluation setting
```
- summarize(self):
```
s = self.eval['precision']
t = np.where(iouThr == p.iouThrs)[0]
s = s[t]
s = s[:, :, :, aind, mind]
mean_s = np.mean(s[s > -1])
```
