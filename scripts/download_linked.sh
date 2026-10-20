#!/usr/bin/env bash
# NCHS linked birth/infant death files (period numerators) 2015-2024 and their user guides.
set -uo pipefail
mkdir -p "$(dirname "$0")/../inputs/linked"; cd "$(dirname "$0")/../inputs/linked"
B=https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Datasets/DVS
D=https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Dataset_Documentation/DVS
{ echo "$B/periodlinkedus/LinkPE15US.zip LinkPE15US.zip"; echo "$B/periodlinkedus/LinkPE16US.zip LinkPE16US.zip"
  for y in 2017 2018 2019 2020 2021 2022 2023 2024; do echo "$B/period-cohort-linked/${y}PE$((y-1))CO.zip ${y}PE$((y-1))CO.zip"; done
  echo "$D/periodlinked/LinkPE15Guide.pdf guide_PE15.pdf"; echo "$D/periodlinked/LinkPE16Guide.pdf guide_PE16.pdf"
  for y in 17 18 19 20 21 22 23 24; do echo "$D/period-cohort-linked/${y}PE$((y-1))CO_linkedUG.pdf guide_${y}.pdf"; done
} | xargs -P 12 -n 2 sh -c "curl -sfL --retry 5 -C - -o \"\$1\" \"\$0\" || echo FAIL \$0"
ls -la
