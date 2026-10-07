# Identical 24-ID teacher-prefix conditioning panel

Every condition below uses the exact same 24 ordinary examples within a checkpoint and across checkpoints. CE is token-weighted; early agreement uses first 1/8 positions, full-prefix agreement is conditioned on the saved teacher targets. These are not free-running content/emotion accuracy. Wrong audio is resampled to equal state length; feature statistics change. TEST results are descriptive, never selection inputs.

| Method | Condition | n | Tokens | CE | First1 | First8 | Full-prefix |
|---|---|---:|---:|---:|---:|---:|---:|
| evaluation_epoch1 | speech | 24 | 5245 | 1.622393 | 0.7917 | 0.7552 | 0.7897 |
| evaluation_epoch1 | shuffled_speech | 24 | 5245 | 1.793592 | 0.3750 | 0.5208 | 0.7535 |
| evaluation_epoch1 | zero_speech | 24 | 5245 | 1.888764 | 0.2083 | 0.4219 | 0.7907 |
| evaluation_ce_control | speech | 24 | 5245 | 1.614342 | 0.9167 | 0.7760 | 0.7872 |
| evaluation_ce_control | shuffled_speech | 24 | 5245 | 1.811697 | 0.5000 | 0.5260 | 0.7455 |
| evaluation_ce_control | zero_speech | 24 | 5245 | 1.888764 | 0.2083 | 0.4219 | 0.7907 |
| evaluation_transcript30 | speech | 24 | 5245 | 1.618734 | 0.8333 | 0.7604 | 0.7840 |
| evaluation_transcript30 | shuffled_speech | 24 | 5245 | 1.816747 | 0.4583 | 0.5208 | 0.7459 |
| evaluation_transcript30 | zero_speech | 24 | 5245 | 1.888817 | 0.2083 | 0.4271 | 0.7905 |
| evaluation_ordinarykl | speech | 24 | 5245 | 1.675309 | 0.8750 | 0.7604 | 0.8698 |
| evaluation_ordinarykl | shuffled_speech | 24 | 5245 | 1.859876 | 0.5000 | 0.5469 | 0.8088 |
| evaluation_ordinarykl | zero_speech | 24 | 5245 | 1.888764 | 0.2083 | 0.4219 | 0.7907 |
