# Board learning under distribution shift (leave-one-cluster-out)

Hold out one spin cluster entirely; train on the other two (114 serves total). Gate: Mahalanobis distance over the standardized features vs chi2(q=0.999, df=9). Bounce-fit: least-squares over (e, mu, alpha) through the flight->bounce->flight map on the training labels.

| held-out cluster | split | physics | ridge | ridge+gate | bounce-fit | gate fallback rate |
|---|---|---|---|---|---|---|
| topspin | ID (18) | 17.1 | 3.6 | 3.6 | 4.1 | 0% |
| topspin | OOD (40) | 26.4 | 11.6 | 21.4 | 12.9 | 68% |
| flat | ID (20) | 22.9 | 3.4 | 4.1 | 7.6 | 5% |
| flat | OOD (34) | 19.6 | 6.3 | 6.8 | 7.7 | 6% |
| backspin | ID (18) | 22.8 | 2.8 | 2.8 | 7.6 | 0% |
| backspin | OOD (40) | 15.1 | 7.3 | 8.1 | 4.2 | 8% |

fitted bounce params per fold (default e=0.7765 mu=0.25 alpha=0.4):
- held-out topspin: e=0.689  mu=0.050  alpha=0.200
- held-out flat: e=0.793  mu=0.050  alpha=0.200
- held-out backspin: e=0.779  mu=0.050  alpha=0.200
