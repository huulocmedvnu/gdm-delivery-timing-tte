#!/usr/bin/env bash
# Full pipeline from the NCHS public-use files to outputs/ and figures/.
# PY: Python with requirements.txt. NCPU: parallel file jobs. SEVENZIP: 7-Zip binary (two NCHS zips use PPMd).
# Set SKIP_DOWNLOAD=1 if inputs/ is already filled.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}; NCPU=${NCPU:-12}; SEVENZIP=${SEVENZIP:-7zz}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p work outputs figures
[ -n "${SKIP_DOWNLOAD:-}" ] || { bash scripts/download.sh; bash scripts/download_linked.sh; }
for c in rust rust_ccw; do (cd $c && cargo build --release -q) || exit 1; done
export V=rust/target/release/vitals SEVENZIP

# unzip one file to stdout; natality 2015 and fetal death 2016 are PPMd zips that need 7-Zip
unz() { if unzip -p "$1" 2>/dev/null | head -c 1 | grep -q .; then unzip -p "$1"; else "$SEVENZIP" e -so "$1" 2>/dev/null; fi; }
export -f unz

# 1. birthweight 10th percentiles by week and sex (small for gestational age), natality 2019-2024
mkdir -p work/agg_bwq
ls inputs/natality/Nat20{19,20,21,22,23,24}us.zip | xargs -P "$NCPU" -I{} bash -c 'unz {} | $V nat 8 bwq > work/agg_bwq/$(basename {} .zip).tsv 2>/dev/null'
cat work/agg_bwq/*.tsv | grep ^BWQ > work/bwq.tsv
$PY scripts/bw_thresholds.py > /dev/null
export BWQ=work/bw_p10.tsv

# 2. covariate-pattern counts for the gestational diabetes cohort, 2015-2024, with linked neonatal deaths
mkdir -p work/agg_gdm; rm -f work/agg_gdm/*
{ ls inputs/natality/*.zip inputs/fetal/*.zip | sed "s/^/F /"
  printf "L %s\n" "LinkPE15US.zip:VS15LINK.USNUMPUB" "LinkPE16US.zip:VS16LINK.Public.USNUMPUB_2019_08_26" \
  "2017PE2016CO.zip:VS17LINK.Public.USNUMPUB_R2019_05_20" "2018PE2017CO.zip:VS18LINK.Public.USNUMPUB_R2020_04_03" \
  "2019PE2018CO.zip:VS19LINK.Public.USNUMPUB_R2021_08_30" "2020PE2019CO.zip:VS20LINK.Public.USNUMPUB_R2022_05_03" \
  "2021PE2020CO.zip:VS2021LINK.Public.USNUMPUB_R2023_05_31" "2022PE2021CO.zip:VS2022LINK.Public.USNUMPUB_R2024_08_07" \
  "2023PE2022CO.zip:VS2023LINK.Public.USNUMPUB_R2024_11_27" "2024PE2023CO.zip:VS2024LINK.Public.USNUMPUB_R2026_01_14"; } |
xargs -P "$NCPU" -L 1 bash -c 'if [ "$0" = F ]; then k=nat; case $1 in *Fetal*) k=fet;; esac
  unz "$1" | $V $k 8 gdm > work/agg_gdm/$(basename "$1" .zip).tsv 2>/dev/null
else z=${1%%:*}; m=${1#*:}; unzip -p inputs/linked/$z "$m" | $V inf 4 gdm > work/agg_gdm/$m.tsv 2>/dev/null; fi'
cat work/agg_gdm/*.tsv > work/counts_gdm.tsv

# 3. study flow for the 39-week trial, 2019-2024
mkdir -p work/agg_flow
ls inputs/natality/Nat20{19,20,21,22,23,24}us.zip inputs/fetal/Fetal20{19,20,21,22,23,24}US.zip |
  xargs -P "$NCPU" -I{} bash -c 'k=nat; case {} in *Fetal*) k=fet;; esac; unz {} | $V $k 8 gdmflow > work/agg_flow/$(basename {} .zip).tsv 2>/dev/null'
cat work/agg_flow/*.tsv | awk -F'\t' '{n[$4]+=$5} END{for(s in n) print s"\t"n[s]}' | sort -n > outputs/flow_counts.tsv

# 4. clone-censor-weight analysis, trials at 39, 40 and 38 weeks (refitted Bayesian bootstrap in rust_ccw)
(cd scripts && $PY -W ignore gdm_ccw.py main)
# 5. propensity score matched conventional comparison (post hoc amendment a)
(cd scripts && $PY -W ignore psm.py 1000 > /dev/null)
# 6. reported numbers, then figures and the balance table
(cd scripts && $PY -W ignore manuscript_numbers.py > /dev/null && $PY -W ignore figures.py > /dev/null)
echo "done: outputs/ and figures/"
