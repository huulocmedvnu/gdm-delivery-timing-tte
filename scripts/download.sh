#!/usr/bin/env bash
# CDC NCHS public-use natality (US) and fetal death (US) files, 2015-2024, plus user guides. Parallel.
set -uo pipefail
mkdir -p "$(dirname "$0")/../inputs/natality" "$(dirname "$0")/../inputs/fetal" "$(dirname "$0")/../inputs/docs"; cd "$(dirname "$0")/../inputs"
B=https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Datasets/DVS
D=https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Dataset_Documentation/DVS
U=$(mktemp); trap "rm -f $U" EXIT
for y in 2015 2016 2017 2018 2019 2020 2021 2022 2023 2024; do
  case $y in 2017) n=Nat2017US.zip;; 2021|2022) n=nat${y}us.zip;; *) n=Nat${y}us.zip;; esac
  echo "$B/natality/$n natality/Nat${y}us.zip"
  if [ $y -le 2017 ]; then f=Fetal${y}US.zip; else f=Fetal${y}US_COD.zip; fi
  echo "$B/fetaldeathus/$f fetal/Fetal${y}US.zip"
done > "$U"
cat "$U" | xargs -P 20 -n 2 sh -c "curl -sfL --retry 5 -C - -o \"\$1\" \"\$0\" || echo FAIL \$0"
for y in 2015 2016 2017 2018 2019; do curl -sfL -o docs/fetal_${y}.pdf $D/fetaldeath/${y}FetalUserGuide.pdf; done
for y in 2020 2021 2022 2023 2024; do curl -sfL -o docs/fetal_${y}.pdf $D/fetaldeath/${y}fetaluserguide.pdf; done
for f in UserGuide2015.pdf UserGuide2016.pdf UserGuide2017.pdf UserGuide2018.pdf UserGuide2019-508.pdf UserGuide2020.pdf UserGuide2021.pdf UserGuide2022.pdf UserGuide2023.pdf UserGuide2024.pdf; do curl -sfL -o docs/nat_$f $D/natality/$f || echo "nodoc $f"; done
ls -la natality fetal docs
