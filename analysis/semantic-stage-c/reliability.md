# Stage C Semantic Judge Reliability

This report is outcome-blind. It summarizes measurement availability, retries, validation failures, judge coverage, consensus status, and resume provenance only.

## Coverage

- Effective trajectories: 100
- Core judge jobs: 200
- Valid judge jobs: 199
- Missing judge jobs: 1
- Judge-job coverage: 99.50%

## Retry behavior

- Provider attempts: 209
- Excess attempts: 9
- Jobs requiring retry: 7 (3.50%)
- Attempt-count distribution: {'1': 193, '2': 5, '3': 2}

## Core-judge consensus

- Field comparisons: 500
- Agreement: 471
- Disagreement: 24
- Missing: 5
- Raw agreement among evaluable fields: 95.15%
- Missing-field rate: 1.00%

## Interpretation

Stage C is operationally suitable for production-scale semantic measurement. Terminal missingness is retained as missing rather than converted to a negative label or retried beyond the frozen attempt budget.
