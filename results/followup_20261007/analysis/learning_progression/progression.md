# Fixed-corpus10Hz response-CE progression

9550 extension rejected; final selection retains the 6775 parent.
Fixed 38,193-example available training pool; updates/exposures change, not dataset size. One seed; validation CE uncertainty is unmeasured.
LR 1e-3 through 4775, then 2e-4; objective remains response CE. Repeated exposures and LR changes confound a pure training-duration effect.
No TEST result or unlike-objective training loss appears in this figure.

| Updates | Exposures | LR | Ordinary CE | Old CE | Neu CE | Macro CE |
|---:|---:|---:|---:|---:|---:|---:|
| 2000 | 16000 | 0.001 | 1.638227 | 0.966929 | 0.845394 | 1.150183 |
| 4775 | 38193 | 0.001 | 1.598574 | 0.819291 | 0.723613 | 1.047159 |
| 6775 | 54193 | 0.0002 | 1.588295 | 0.765527 | 0.639283 | 0.997702 |
| 9550 | 76386 | 0.0002 | 1.580712 | 0.749637 | 0.634656 | 0.988335 |