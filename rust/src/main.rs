//! Aggregates NCHS natality or fetal death public-use records streamed on stdin (fixed width).
//! vitals <nat|fet> <threads> [ext]
//!   default: kind, year, ga, bmi_x10, age_grp, nullip, race, pdiab, phyp, induct, cesar, estofd, n
//!   ext:     kind, year, month, ga, bmi_class, age_grp, nullip, race, pdiab, phyp, gpreg, induct, cesar, estofd,
//!            n, nicu, apgar_lt7, avent, seizure, transfusion, rupture, icu
//!   inf (linked infant death numerators, natality layout + AGED 1356-1358, DOD_YY 1672-1675):
//!            inf, death_year, ga, bmi_class, age_grp, nullip, race, pdiab, phyp, neonatal_lt28d, early_lt7d
//!   dec  (any kind, weeks 37-44): kind, year, ga, bmi_int (capped 60), nullip, race, lowrisk, anomaly, estofd,
//!            n, nicu, cesarean, morbidity, neonatal_lt28d   (lowrisk: no diabetes, hypertensive disorder, previous
//!            caesarean, cephalic; anomaly: natality items 537-542, 549-553, fetal IICOD Q*, infant UCOD Q*)
//!   tte  (weeks 39-44, any kind): kind, year, 0, ga, bmi_class, age_grp, nullip, race, lowrisk, anomaly, educ,
//!            induct, 0, estofd, n, nicu, cesarean, morbidity, neonatal_lt28d (educ: 1 HS or less, 2 some college,
//!            3 bachelor+, 9 unknown; MEDUC natality 124, fetal 145)
//!   ps   (2019+, weeks 39-44, low-risk non-anomalous only): kind, year, ga, bmi_class, bmi_int, age6, nullip, race, educ,
//!            precare, wic, smoke, height, inftr, induct, payer, married, trial, sga, estofd, n, nicu, cesarean, morbidity, nd,
//!            apgar<7, vent_now, vent_6h, surfactant, antibiotics, seizures, transfusion, laceration34, rupture, hysterectomy, icu, smm
//!   gdm  (2015+, weeks 38-44, GDM cohort: gestational diabetes, no prepregnancy diabetes or chronic hypertension, no previous
//!            cesarean, cephalic, non-anomalous): ps columns plus ghyp (gestational hypertension or eclampsia) after sga, and
//!            birthweight >= 4500 g as the last count
//!   bwq  (natality only): histogram of birthweight (10 g bins) by completed week and sex for the ps cohort
//! Validation lines (stderr): VAL kind year resident_all fetal_20plus. Positions are 1-based, from the NCHS guides.
use std::collections::HashMap;
use std::io::{self, Read, Write};

#[derive(Clone, Copy)]
struct Pos { year: (usize, usize), month: (usize, usize), age: (usize, usize), res: usize, race: usize,
             plive: (usize, usize), pdead: (usize, usize), bmi: (usize, usize), pdiab: usize, gdiab: usize,
             phyp: usize, ghyp: usize, ecl: usize, plur: usize, ga: (usize, usize), induct: Option<usize>,
             route: usize, estofd: Option<usize>, nat: bool }

const NAT: Pos = Pos { year: (9, 12), month: (13, 14), age: (75, 76), res: 104, race: 117, plive: (171, 172),
    pdead: (173, 174), bmi: (283, 286), pdiab: 313, gdiab: 314, phyp: 315, ghyp: 316, ecl: 317, plur: 454,
    ga: (499, 500), induct: Some(383), route: 402, estofd: None, nat: true };
const FET: Pos = Pos { year: (11, 14), month: (15, 16), age: (86, 87), res: 131, race: 144, plive: (181, 182),
    pdead: (183, 184), bmi: (245, 248), pdiab: 257, gdiab: 258, phyp: 259, ghyp: 260, ecl: 261, plur: 301,
    ga: (340, 341), induct: None, route: 275, estofd: Some(357), nat: false };

fn field(r: &[u8], p: (usize, usize)) -> &[u8] { if r.len() >= p.1 { &r[p.0 - 1..p.1] } else { b"" } }
fn ch(r: &[u8], p: usize) -> u8 { if r.len() >= p { r[p - 1] } else { b' ' } }
fn int(b: &[u8]) -> Option<i32> { std::str::from_utf8(b).ok()?.trim().parse().ok() }
fn num(b: &[u8]) -> Option<f32> { std::str::from_utf8(b).ok()?.trim().parse().ok() }
fn yes(r: &[u8], p: usize) -> u8 { (ch(r, p) == b'Y') as u8 }

pub fn bmi_class(b10: u16) -> u8 {
    match b10 { 0..=184 => 0, 185..=249 => 1, 250..=299 => 2, 300..=349 => 3, 350..=399 => 4, 400..=499 => 5, _ => 6 }
}

type K0 = (u16, u8, u16, u8, u8, u8, u8, u8, u8, u8, u8);
type K1 = ((u16, u8, u8, u8, u8, u8, u8), (u8, u8, u8, u8, u8, u8));

struct Common { year: u16, month: u8, ga: u8, b10: u16, agegrp: u8, nullip: u8, race: u8 }

fn common(r: &[u8], p: &Pos) -> Option<Common> {
    if ch(r, p.plur) != b'1' { return None; }
    if !(b'1'..=b'3').contains(&ch(r, p.res)) { return None; }
    let ga = match int(field(r, p.ga)) { Some(g) if (20..=44).contains(&g) => g as u8, _ => return None };
    let b10 = match num(field(r, p.bmi)) { Some(b) if b > 0.0 && b < 99.0 => (b * 10.0).round() as u16, _ => return None };
    let age = int(field(r, p.age)).unwrap_or(0);
    let nullip = match (int(field(r, p.plive)), int(field(r, p.pdead))) {
        (Some(a), Some(b)) if a < 99 && b < 99 => if a + b == 0 { 1 } else { 0 }, _ => 9 };
    let race = match ch(r, p.race) { b'1' => 1, b'2' => 2, b'4' => 4, b'7' => 7, b'8' => 8, b'3' | b'5' | b'6' => 3, _ => 9 };
    Some(Common { year: int(field(r, p.year)).unwrap_or(0) as u16, month: int(field(r, p.month)).unwrap_or(0) as u8,
                  ga, b10, agegrp: if age < 25 { 0 } else if age < 35 { 1 } else { 2 }, nullip, race })
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let p = if a[1] == "fet" { FET } else { NAT };
    let inf = a[1] == "inf";
    let threads: usize = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(8);
    let ext = a.get(3).map(|s| s == "ext" || s == "dec").unwrap_or(false);
    let gdm = a.get(3).map(|s| s == "gdm").unwrap_or(false);
    let ps = a.get(3).map(|s| s == "ps" || s == "bwq" || s == "gdm").unwrap_or(false);
    let flow = a.get(3).map(|s| s == "flow" || s == "gdmflow").unwrap_or(false);
    let gdmflow = a.get(3).map(|s| s == "gdmflow").unwrap_or(false);
    let bwq = a.get(3).map(|s| s == "bwq").unwrap_or(false);
    // 10th percentile birthweight thresholds by (completed week, sex), file from env BWQ: "ga<TAB>sex<TAB>grams"
    let mut thr: HashMap<(u8, u8), i32> = HashMap::new();
    if let Ok(f) = std::env::var("BWQ") {
        for l in std::fs::read_to_string(f).unwrap_or_default().lines() {
            let v: Vec<&str> = l.split('\t').collect();
            if v.len() == 3 { if let (Ok(g), Ok(x)) = (v[0].parse::<u8>(), v[2].parse::<i32>()) { thr.insert((g, v[1].as_bytes()[0]), x); } }
        }
    }
    let thr = &thr;
    let tte = a.get(3).map(|s| s == "tte" || s == "ps" || s == "bwq" || s == "gdm").unwrap_or(false);
    let dec = a.get(3).map(|s| s == "dec" || s == "tte" || s == "ps" || s == "bwq" || s == "gdm").unwrap_or(false);
    let kind: &str = a[1].as_str();
    let mut buf = Vec::new(); io::stdin().lock().read_to_end(&mut buf).unwrap();
    let mut cuts = vec![0usize];
    for t in 1..threads { let mut c = buf.len() * t / threads; while c < buf.len() && buf[c] != b'\n' { c += 1; } cuts.push((c + 1).min(buf.len())); }
    cuts.push(buf.len());
    type Out = (HashMap<K0, u64>, HashMap<K1, [u64; 8]>, HashMap<u16, (u64, u64)>, HashMap<String, [u64; 18]>);
    let parts: Vec<Out> = std::thread::scope(|s| {
        let hs: Vec<_> = (0..threads).map(|t| { let chunk = &buf[cuts[t]..cuts[t + 1]]; s.spawn(move || {
            let (mut m0, mut m1, mut val, mut psm): Out = (HashMap::new(), HashMap::new(), HashMap::new(), HashMap::new());
            for r in chunk.split(|&b| b == b'\n') {
                if r.len() < 20 { continue; }
                if (b'1'..=b'3').contains(&ch(r, p.res)) {
                    let y = int(field(r, p.year)).unwrap_or(0) as u16; let e = val.entry(y).or_insert((0, 0));
                    e.0 += 1; if !p.nat { if let Some(g) = int(field(r, p.ga)) { if (20..=47).contains(&g) { e.1 += 1; } } }
                }
                if flow {
                    if !(b'1'..=b'3').contains(&ch(r, p.res)) { continue; }
                    let year = int(field(r, p.year)).unwrap_or(0);
                    if !(2019..=2024).contains(&year) { continue; }
                    let gar = int(field(r, p.ga));
                    if !p.nat && !matches!(gar, Some(g) if (20..=47).contains(&g)) { continue; }
                    let (pres, prevc) = if p.nat { (401, 331) } else { (274, 265) };
                    let ca_nat = || [537usize, 538, 539, 540, 541, 542, 549, 550, 551].iter().any(|&i| ch(r, i) == b'Y')
                        || [552usize, 553].iter().any(|&i| ch(r, i) == b'C' || ch(r, i) == b'P');
                    let anom_f = || if p.nat { ca_nat() } else { ch(r, 2603) == b'Q' };
                    // gdmflow (39-week trial): 1 multiple, 2 gestation <39 or unknown, 3 BMI unknown, 4 no GDM, 5 prepregnancy
                    // diabetes or chronic hypertension, 6 previous cesarean or noncephalic or unknown presentation, 7 anomaly, 8 cohort
                    let step: u8 = if gdmflow { if ch(r, p.plur) != b'1' { 1 }
                        else if !matches!(gar, Some(g) if (39..=44).contains(&g)) { 2 }
                        else if !matches!(num(field(r, p.bmi)), Some(b) if b > 0.0 && b < 99.0) { 3 }
                        else if yes(r, p.gdiab) == 0 { 4 }
                        else if (yes(r, p.pdiab) | yes(r, p.phyp)) != 0 { 5 }
                        else if yes(r, prevc) != 0 || ch(r, pres) != b'1' { 6 }
                        else if anom_f() { 7 } else { 8 } } else
                    if ch(r, p.plur) != b'1' { 1 }
                        else if !matches!(gar, Some(g) if (39..=44).contains(&g)) { 2 }
                        else if !matches!(num(field(r, p.bmi)), Some(b) if b > 0.0 && b < 99.0) { 3 }
                        else if (yes(r, p.pdiab) | yes(r, p.gdiab) | yes(r, p.phyp) | yes(r, p.ghyp) | yes(r, p.ecl) | yes(r, prevc)) != 0 || ch(r, pres) != b'1' { 4 }
                        else if (if p.nat { ca_nat() } else { ch(r, 2603) == b'Q' }) { 5 } else { 6 };
                    let e = psm.entry(format!("FLOW\t{}\t{}\t{}", kind, year, step)).or_insert([0; 18]); e[0] += 1;
                    continue;
                }
                let c = match common(r, &p) { Some(c) => c, None => continue };
                if dec {
                    if c.ga < (if gdm { 38 } else if tte { 39 } else { 37 }) { continue; }
                    let (pres, prevc) = if p.nat { (401, 331) } else { (274, 265) };
                    let low = (yes(r, p.pdiab) | yes(r, p.gdiab) | yes(r, p.phyp) | yes(r, p.ghyp) | yes(r, p.ecl) | yes(r, prevc)) == 0 && ch(r, pres) == b'1';
                    let ca_nat = || [537usize, 538, 539, 540, 541, 542, 549, 550, 551].iter().any(|&i| ch(r, i) == b'Y')
                        || [552usize, 553].iter().any(|&i| ch(r, i) == b'C' || ch(r, i) == b'P');
                    let anom = if inf { ca_nat() || ch(r, 1368) == b'Q' } else if p.nat { ca_nat() } else { ch(r, 2603) == b'Q' };
                    let mut v = [1u64, 0, 0, 0, 0, 0, 0, 0];
                    let mut year = c.year;
                    if inf {
                        let aged = int(field(r, (1356, 1358))).unwrap_or(999); if aged > 27 { continue; }
                        year = int(field(r, (1672, 1675))).unwrap_or(0) as u16; v = [0, 0, 0, 0, 1, 0, 0, 0];
                    } else if p.nat {
                        v[1] = (ch(r, 519) == b'Y') as u64;
                        v[2] = matches!(ch(r, p.route), b'3' | b'4') as u64;
                        let apg = matches!(int(field(r, (444, 445))), Some(x) if x < 7);
                        v[3] = (apg || ch(r, 517) == b'Y' || ch(r, 518) == b'Y' || ch(r, 522) == b'Y') as u64;
                    }
                    let est = p.estofd.map(|i| ch(r, i)).unwrap_or(b'-');
                    let bi = ((c.b10 / 10).min(60)) as u8;
                    if ps {
                        let gdm_ok = yes(r, p.gdiab) == 1 && yes(r, p.pdiab) == 0 && yes(r, p.phyp) == 0 && yes(r, prevc) == 0 && ch(r, pres) == b'1';
                        if anom || (gdm && (year < 2015 || !gdm_ok)) || (!gdm && (year < 2019 || !low)) { continue; }
                        let n = |nat: usize, fet: usize| if p.nat { nat } else { fet };
                        let age = int(field(r, p.age)).unwrap_or(0);
                        let age6: u8 = match age { 0..=19 => 0, 20..=24 => 1, 25..=29 => 2, 30..=34 => 3, 35..=39 => 4, _ => 5 };
                        let educ = match ch(r, n(124, 145)) { b'1' | b'2' | b'3' => 1u8, b'4' | b'5' => 2, b'6' | b'7' | b'8' => 3, _ => 9 };
                        let pc = int(field(r, if p.nat { (224, 225) } else { (202, 203) })).unwrap_or(99);
                        let precare: u8 = match pc { 1..=3 => 1, 4..=6 => 2, 7..=10 => 3, 0 => 4, _ => 9 };
                        let wic = match ch(r, n(251, 229)) { b'Y' => 1u8, b'N' => 0, _ => 9 };
                        let smoke = match ch(r, n(269, 238)) { b'Y' => 1u8, b'N' => 0, _ => 9 };
                        let ht = int(field(r, if p.nat { (280, 281) } else { (243, 244) })).unwrap_or(99);
                        let height: u8 = match ht { 30..=59 => 0, 60..=63 => 1, 64..=67 => 2, 68..=78 => 3, _ => 9 };
                        let inftr = (ch(r, n(325, 262)) == b'Y') as u8;
                        let ind = if p.nat { match ch(r, 383) { b'Y' => 1u8, b'N' => 0, _ => 9 } } else { 8 };
                        let payer = if p.nat { match ch(r, 436) { b'1' => 1u8, b'2' => 2, b'3' => 3, b'4' => 4, _ => 9 } } else { 8 };
                        let married = if p.nat { match ch(r, 120) { b'1' => 1u8, b'2' => 2, _ => 9 } } else { 8 };
                        let trial = if p.nat { match ch(r, 403) { b'Y' => 1u8, b'N' => 0, b'X' => 2, _ => 9 } } else { 8 };
                        let bw = int(field(r, if p.nat { (504, 507) } else { (349, 352) })).unwrap_or(9999);
                        let sex = ch(r, if p.nat { 475 } else { 316 });
                        if bwq {
                            if !p.nat || inf || bw >= 9999 { continue; }
                            let e = psm.entry(format!("BWQ\t{}\t{}\t{}", c.ga, sex as char, bw / 10 * 10)).or_insert([0; 18]); e[0] += 1;
                            continue;
                        }
                        let sga: u8 = match thr.get(&(c.ga, sex)) { Some(&t) if bw < 9999 => (bw < t) as u8, _ => 9 };
                        let mut line = format!("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}", kind, year, c.ga, bmi_class(c.b10), bi, age6, c.nullip, c.race,
                                           educ, precare, wic, smoke, height, inftr, ind, payer, married, trial, sga);
                        if gdm { line.push_str(&format!("\t{}", (yes(r, p.ghyp) | yes(r, p.ecl)))); }
                        line.push_str(&format!("\t{}", est as char));
                        let e = psm.entry(line).or_insert([0; 18]); for i in 0..5 { e[i] += v[i]; }
                        if p.nat && !inf {
                            // safety outcomes among live births: apgar<7, ventilation now / >6 h, surfactant, antibiotics, seizures,
                            // transfusion, 3rd/4th degree laceration, ruptured uterus, unplanned hysterectomy, ICU, severe maternal composite
                            let y = |i: usize| (ch(r, i) == b'Y') as u64;
                            let apg = matches!(int(field(r, (444, 445))), Some(x) if x < 7) as u64;
                            let mm = [y(415), y(416), y(417), y(418), y(419)];
                            let smm = (mm[0] | mm[2] | mm[3] | mm[4]) as u64;
                            let xs = [apg, y(517), y(518), y(520), y(521), y(522), mm[0], mm[1], mm[2], mm[3], mm[4], smm];
                            for (j, x) in xs.iter().enumerate() { e[5 + j] += x; }
                            e[17] += (bw >= 4500 && bw < 9999) as u64;
                        }
                        continue;
                    }
                    let k = if tte {
                        let educ = match ch(r, if p.nat { 124 } else { 145 }) { b'1' | b'2' | b'3' => 1u8, b'4' | b'5' => 2, b'6' | b'7' | b'8' => 3, _ => 9 };
                        let ind = if p.nat { match ch(r, 383) { b'Y' => 1u8, b'N' => 0, _ => 9 } } else { 8 };
                        ((year, 0u8, c.ga, bmi_class(c.b10), c.agegrp, c.nullip, c.race), (low as u8, anom as u8, educ, ind, 0u8, est))
                    } else { ((year, 0u8, c.ga, bi, 0u8, c.nullip, c.race), (low as u8, anom as u8, 0u8, 0u8, 0u8, est)) };
                    let e = m1.entry(k).or_insert([0; 8]); for i in 0..8 { e[i] += v[i]; }
                    continue;
                }
                if inf {
                    let aged = int(field(r, (1356, 1358))).unwrap_or(999); let dy = int(field(r, (1672, 1675))).unwrap_or(0) as u16;
                    if aged > 27 { continue; }
                    let k = ((dy, 0u8, c.ga, bmi_class(c.b10), c.agegrp, c.nullip, c.race), (yes(r, p.pdiab), yes(r, p.phyp), 0u8, 0u8, 0u8, b'-'));
                    let e = m1.entry(k).or_insert([0; 8]); e[0] += 1; if aged < 7 { e[1] += 1; }
                    continue;
                }
                let pdiab = yes(r, p.pdiab); let phyp = yes(r, p.phyp);
                let induct = p.induct.map(|i| match ch(r, i) { b'Y' => 1, b'N' => 0, _ => 9 }).unwrap_or(8);
                let cesar = match ch(r, p.route) { b'3' | b'4' => 1, b'1' | b'2' => 0, _ => 9 };
                let est = p.estofd.map(|i| ch(r, i)).unwrap_or(b'-');
                if !ext {
                    *m0.entry((c.year, c.ga, c.b10, c.agegrp, c.nullip, c.race, pdiab, phyp, induct, cesar, est)).or_insert(0) += 1;
                } else {
                    let gpreg = (yes(r, p.gdiab) | yes(r, p.ghyp) | yes(r, p.ecl)) as u8;
                    let mut v = [1u64, 0, 0, 0, 0, 0, 0, 0];
                    if p.nat {
                        v[1] = (ch(r, 519) == b'Y') as u64;
                        v[2] = match int(field(r, (444, 445))) { Some(x) if x < 7 => 1, _ => 0 };
                        v[3] = (ch(r, 517) == b'Y' || ch(r, 518) == b'Y') as u64;
                        v[4] = (ch(r, 522) == b'Y') as u64;
                        v[5] = (ch(r, 415) == b'Y') as u64; v[6] = (ch(r, 417) == b'Y') as u64; v[7] = (ch(r, 419) == b'Y') as u64;
                    }
                    let k = ((c.year, c.month, c.ga, bmi_class(c.b10), c.agegrp, c.nullip, c.race), (pdiab, phyp, gpreg, induct, cesar, est));
                    let e = m1.entry(k).or_insert([0; 8]); for i in 0..8 { e[i] += v[i]; }
                }
            }
            (m0, m1, val, psm) }) }).collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let (mut m0, mut m1, mut val, mut psm): Out = (HashMap::new(), HashMap::new(), HashMap::new(), HashMap::new());
    for (a0, a1, v, ap) in parts {
        for (k, x) in ap { let e = psm.entry(k).or_insert([0; 18]); for i in 0..18 { e[i] += x[i]; } }
        for (k, n) in a0 { *m0.entry(k).or_insert(0) += n; }
        for (k, x) in a1 { let e = m1.entry(k).or_insert([0; 8]); for i in 0..8 { e[i] += x[i]; } }
        for (y, (r, f)) in v { let e = val.entry(y).or_insert((0, 0)); e.0 += r; e.1 += f; }
    }
    let out = io::stdout(); let mut w = io::BufWriter::new(out.lock());
    for (k, n) in &m0 { writeln!(w, "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}", a[1], k.0, k.1, k.2, k.3, k.4, k.5, k.6, k.7, k.8, k.9, k.10 as char, n).unwrap(); }
    for (k, v) in &m1 {
        let (x, z) = k;
        writeln!(w, "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}", a[1], x.0, x.1, x.2, x.3, x.4, x.5, x.6, z.0,
                 z.1, z.2, z.3, z.4, z.5 as char, v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]).unwrap();
    }
    for (k, x) in &psm { let v: Vec<String> = x.iter().map(|z| z.to_string()).collect(); writeln!(w, "{}\t{}", k, v.join("\t")).unwrap(); }
    for (y, (r, f)) in &val { eprintln!("VAL\t{}\t{}\t{}\t{}", a[1], y, r, f); }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn rec(len: usize, sets: &[(usize, &str)]) -> Vec<u8> {
        let mut r = vec![b' '; len];
        for (pos, s) in sets { for (i, c) in s.bytes().enumerate() { r[pos - 1 + i] = c; } }
        r
    }
    #[test]
    fn natality_fields() {
        let r = rec(1330, &[(9, "2019"), (13, "08"), (75, "31"), (104, "1"), (117, "2"), (171, "00"), (173, "00"),
                            (283, "41.3"), (313, "N"), (315, "Y"), (454, "1"), (499, "39")]);
        let c = common(&r, &NAT).unwrap();
        assert_eq!((c.year, c.month, c.ga, c.b10, c.agegrp, c.nullip, c.race), (2019, 8, 39, 413, 1, 1, 2));
        assert_eq!(bmi_class(c.b10), 5); assert_eq!(yes(&r, 315), 1);
    }
    #[test]
    fn exclusions() {
        let twin = rec(1330, &[(104, "1"), (454, "2"), (499, "39"), (283, "25.0")]); assert!(common(&twin, &NAT).is_none());
        let foreign = rec(1330, &[(104, "4"), (454, "1"), (499, "39"), (283, "25.0")]); assert!(common(&foreign, &NAT).is_none());
        let nobmi = rec(1330, &[(104, "1"), (454, "1"), (499, "39"), (283, "99.9")]); assert!(common(&nobmi, &NAT).is_none());
    }
    #[test]
    fn bmi_boundaries() {
        assert_eq!(bmi_class(184), 0); assert_eq!(bmi_class(185), 1); assert_eq!(bmi_class(249), 1); assert_eq!(bmi_class(250), 2);
        assert_eq!(bmi_class(399), 4); assert_eq!(bmi_class(400), 5); assert_eq!(bmi_class(499), 5); assert_eq!(bmi_class(500), 6);
    }
    #[test]
    fn fetal_fields() {
        let r = rec(3150, &[(11, "2021"), (15, "03"), (86, "28"), (131, "2"), (144, "7"), (181, "01"), (183, "00"),
                            (245, "52.0"), (301, "1"), (340, "40"), (357, "N")]);
        let c = common(&r, &FET).unwrap();
        assert_eq!((c.year, c.month, c.ga, c.b10, c.nullip, c.race), (2021, 3, 40, 520, 0, 7)); assert_eq!(bmi_class(c.b10), 6);
    }
}
