# FairFace Label Fix Summary

Generated on 2026-04-27 after fixing local FairFace numeric labels.

## What Changed

- The local FairFace CSV stores numeric labels, but the loader expected strings.
- Age labels now map numeric FairFace age classes into the model's 8 age bins.
- Gender labels now use numeric 0/1 directly.
- Race labels now use the numeric order documented in
  `datasets/datasets/FairFace/README.md`:
  - 0 East Asian -> Asian
  - 1 Indian -> Indian
  - 2 Black -> Black
  - 3 White -> White
  - 4 Middle Eastern -> Others
  - 5 Latino/Hispanic -> Others
  - 6 Southeast Asian -> Asian

## Result

| Task | Dataset | Before | After |
| --- | --- | ---: | ---: |
| age MAE | FairFace | 12.8490 years | 6.2990 years |
| gender accuracy | FairFace | 53.3002% | 93.1121% |
| race accuracy | FairFace | 19.6412% / 13.4517% with wrong numeric order | 64.5883% |

The FairFace race result is now meaningful, but it is still not directly
paper-comparable because the current model head is 5-class while FairFace is
originally 7-class.
