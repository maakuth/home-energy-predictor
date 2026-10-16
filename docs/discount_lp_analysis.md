# Battery LP Discount Analysis (Offline Quality Fixtures)

Full replay evaluation captured on 2026-10-16. Scope is the four included
immutable quality fixtures using `nemotron-linprog`; lower EUR cost is better.
All runs completed with no safety violations.

| Discount | Portfolio EUR | Change vs 0.995 | Jan | May | Jul | Oct |
|---|---:|---:|---:|---:|---:|---:|
| 0.995 | -81.6424 | 0.0000 | -22.1459 | -20.1578 | -32.0772 | -7.2614 |
| 0.996 | -81.5792 | +0.0632 | -22.0635 | -20.1349 | -32.0384 | -7.3424 |
| 0.997 | -82.1928 | -0.5504 | -22.2008 | -20.3581 | -32.1005 | -7.5334 |
| 0.9975 | -82.7785 | -1.1361 | -22.4534 | -20.7621 | -32.0126 | -7.5504 |
| 0.998 | -82.8932 | -1.2508 | -22.4279 | -20.8683 | -32.0211 | -7.5758 |
| 0.9985 | -82.7803 | -1.1379 | -22.4182 | -20.7200 | -31.9787 | -7.6634 |
| 0.999 | -82.2875 | -0.6451 | -22.3624 | -20.6270 | -31.7967 | -7.5014 |
| 0.9995 | -82.2822 | -0.6398 | -22.3661 | -20.6412 | -31.7831 | -7.4919 |

`0.998` is the portfolio optimum in this sweep, but it regresses the July
fixture. `0.997` is the highest evaluated setting that improves every fixture.
The July regression occurs because the less aggressive discount carries more
night-charged energy forward and shifts export from the highest midday-price
period to lower-priced evening periods. It is economic timing, not a safety or
feasibility failure.

## Slack-only variant

The evaluated `BATTERY_LP_DISCOUNT_SLACK_ONLY=True` POC leaves energy terms
undiscounted and discounts only overflow slack and terminal value. It produced
portfolio cost `-81.9191` EUR, including a July regression of `+0.4296` EUR.
It should not be adopted.

## Decision record

Do not change the production default without a reviewed trade-off decision.
Adopting `0.998` requires accepting the July regression and updating the
approved reference/tolerances. `0.997` is the conservative alternative.
