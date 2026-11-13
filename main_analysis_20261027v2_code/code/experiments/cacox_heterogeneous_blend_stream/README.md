# Iteration21 execution v2: bounded GPU pair counts

Original signed scientific protocol, source models, clinical arrays, scoring formula,
17 candidates and fivefold internal selection remain unchanged. This separate folder
replaces only dense temporary pair tensors by exact int64 block accumulation and
uses one worker. See ITERATION21_EXECUTION_V2.md and previous_attempt/PRESERVATION.json.
The earlier RAM-guard stop is part of the study history and is explicitly reported.

Gates: check_stream.py; preflight.py; benchmark.py run. Then runner.py prepare;
runner.py launch. Root completion still requires 25 inner groups, all-fivefold choices
before outer, 100 outer MI units ×18 roles, source CUDA weight replay, full evaluation,
independent reconstruction and visual QA. Zero new model fits.
