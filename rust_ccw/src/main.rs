//! Clone-censor-weight estimation with refitted treatment models and a nonparametric (Poisson) bootstrap (CCW_BOOT=freq).
//! Input directory (written by scripts/ccw.py export): meta.json, cells.bin (f64 columns: cls, pid, course, risk, live,
//! then one event column per outcome), X_<k>.bin (row-major f64 design matrix of class k, npat x p).
//! Usage: ccw point <dir>            -> <dir>/point.json (weights fitted from zero, risks, params, truncation)
//!        ccw draws <dir> <n> <seed> <threads> -> <dir>/draws.bin (f64 [n][outcome][class][2]), refits start at point params
//!        ccw glmtest <dir>          -> <dir>/glmtest.json (point GLM params only, for comparison with statsmodels)
use rayon::prelude::*;
use serde_json::{json, Value};
use std::fs;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
static T_COUNT: AtomicU64 = AtomicU64::new(0); static T_GLM: AtomicU64 = AtomicU64::new(0);
static T_PRED: AtomicU64 = AtomicU64::new(0); static T_QUANT: AtomicU64 = AtomicU64::new(0); static N_IT: AtomicU64 = AtomicU64::new(0);
fn tick(a: &AtomicU64, t: std::time::Instant) { a.fetch_add(t.elapsed().as_nanos() as u64, Relaxed); }

// ---------- random numbers: xoshiro256++, standard normal, gamma (Marsaglia-Tsang), beta ----------
struct Rng { s: [u64; 4], spare: Option<f64> }
impl Rng {
    fn new(seed: u64) -> Self {
        let mut z = seed; let mut s = [0u64; 4];
        for v in s.iter_mut() { z = z.wrapping_add(0x9E3779B97F4A7C15); let mut x = z; x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB); *v = x ^ (x >> 31); }
        Rng { s, spare: None }
    }
    fn next_u64(&mut self) -> u64 {
        let r = (self.s[0].wrapping_add(self.s[3])).rotate_left(23).wrapping_add(self.s[0]);
        let t = self.s[1] << 17; self.s[2] ^= self.s[0]; self.s[3] ^= self.s[1]; self.s[1] ^= self.s[2]; self.s[0] ^= self.s[3];
        self.s[2] ^= t; self.s[3] = self.s[3].rotate_left(45); r
    }
    fn unif(&mut self) -> f64 { ((self.next_u64() >> 11) as f64 + 0.5) * (1.0 / 9007199254740992.0) } // (0,1)
    fn normal(&mut self) -> f64 {
        if let Some(z) = self.spare.take() { return z; }
        loop { let u = 2.0 * self.unif() - 1.0; let v = 2.0 * self.unif() - 1.0; let s = u * u + v * v;
            if s > 0.0 && s < 1.0 { let m = (-2.0 * s.ln() / s).sqrt(); self.spare = Some(v * m); return u * m; } }
    }
    fn gamma(&mut self, a: f64) -> f64 {
        if a <= 0.0 { return 0.0; }
        if a < 1.0 { let g = self.gamma(a + 1.0); return g * self.unif().powf(1.0 / a); }
        let d = a - 1.0 / 3.0; let c = 1.0 / (9.0 * d).sqrt();
        loop {
            let x = self.normal(); let v = 1.0 + c * x; if v <= 0.0 { continue; }
            let v = v * v * v; let u = self.unif();
            if u < 1.0 - 0.0331 * x * x * x * x { return d * v; }
            if u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) { return d * v; }
        }
    }
    fn beta(&mut self, a: f64, b: f64) -> f64 { let x = self.gamma(a); let y = self.gamma(b); if x + y > 0.0 { x / (x + y) } else { 0.5 } }
    /// Poisson(lam): multiplication method below 10, PTRS (Hoermann 1993) above.
    fn poisson(&mut self, lam: f64) -> f64 {
        if lam <= 0.0 { return 0.0; }
        if lam < 10.0 { let l = (-lam).exp(); let mut k = 0.0; let mut p = self.unif(); while p > l { k += 1.0; p *= self.unif(); } return k; }
        let slam = lam.sqrt(); let loglam = lam.ln(); let b = 0.931 + 2.53 * slam; let a = -0.059 + 0.02483 * b;
        let invalpha = 1.1239 + 1.1328 / (b - 3.4); let vr = 0.9277 - 3.6224 / (b - 2.0);
        loop {
            let u = self.unif() - 0.5; let v = self.unif(); let us = 0.5 - u.abs();
            let k = ((2.0 * a / us + b) * u + lam + 0.43).floor();
            if us >= 0.07 && v <= vr { return k; }
            if k < 0.0 || (us < 0.013 && v > us) { continue; }
            if (v * invalpha / (a / (us * us) + b)).ln() <= -lam + k * loglam - ln_gamma(k + 1.0) { return k; }
        }
    }
    /// Binomial(n, p), exact: beta order-statistic splitting (Knuth) down to n <= 64, then Bernoulli sums.
    fn binomial(&mut self, n: f64, p: f64) -> f64 {
        let (mut n, mut p) = (n.round().max(0.0), p.clamp(0.0, 1.0)); let mut k = 0.0;
        if p <= 0.0 || n == 0.0 { return 0.0; } if p >= 1.0 { return n; }
        while n > 64.0 {
            let a = 1.0 + (n / 2.0).floor(); let b = n + 1.0 - a; let x = self.beta(a, b);
            if x >= p { n = a - 1.0; p /= x; } else { k += a; n = b - 1.0; p = (p - x) / (1.0 - x); }
        }
        for _ in 0..(n as usize) { if self.unif() < p { k += 1.0; } }
        k
    }
}

/// ln Gamma(x) for x > 0 (Lanczos, g = 7, n = 9).
fn ln_gamma(x: f64) -> f64 {
    const C: [f64; 9] = [0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059,
        12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
    if x < 0.5 { return (std::f64::consts::PI / (std::f64::consts::PI * x).sin()).ln() - ln_gamma(1.0 - x); }
    let x = x - 1.0; let mut a = C[0]; let t = x + 7.5; for i in 1..9 { a += C[i] / (x + i as f64); }
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
}

/// Bootstrap type from CCW_BOOT: "freq" = nonparametric Poisson bootstrap of pregnancies (amendment b), otherwise Bayesian.
fn freq_boot() -> bool { std::env::var("CCW_BOOT").map(|v| v == "freq").unwrap_or(false) }
fn total_draw(rng: &mut Rng, n: f64, freq: bool) -> f64 { if freq { rng.poisson(n) } else { rng.gamma(n) } }

// ---------- weighted logistic regression by IRLS (same problem as statsmodels GLM Binomial with var_weights) ----------
fn expit(e: f64) -> f64 { if e >= 0.0 { 1.0 / (1.0 + (-e).exp()) } else { let z = e.exp(); z / (1.0 + z) } }

fn cholesky_solve(h: &mut [f64], g: &mut [f64], p: usize) -> bool {
    for j in 0..p {
        let mut s = h[j * p + j]; for k in 0..j { s -= h[j * p + k] * h[j * p + k]; }
        if s <= 0.0 { return false; }
        let d = s.sqrt(); h[j * p + j] = d;
        for i in j + 1..p { let mut s = h[i * p + j]; for k in 0..j { s -= h[i * p + k] * h[j * p + k]; } h[i * p + j] = s / d; }
    }
    for i in 0..p { let mut s = g[i]; for k in 0..i { s -= h[i * p + k] * g[k]; } g[i] = s / h[i * p + i]; }
    for i in (0..p).rev() { let mut s = g[i]; for k in i + 1..p { s -= h[k * p + i] * g[k]; } g[i] = s / h[i * p + i]; }
    true
}

/// y in [0,1], v >= 0 (rows with v == 0 ignored). Sparse rows (CSR). One pass per iteration: deviance of the
/// current beta and the IRLS normal equations are accumulated together. Returns (beta, converged).
fn irls(c: &Class, y: &[f64], v: &[f64], start: Option<&[f64]>) -> (Vec<f64>, bool) {
    let p = c.p; let n = y.len();
    let mut beta = start.map(|s| s.to_vec()).unwrap_or_else(|| vec![0.0; p]);
    let mut h = vec![0.0; p * p]; let mut g = vec![0.0; p]; let mut dold = f64::NAN;
    for _it in 0..101 {
        h.iter_mut().for_each(|z| *z = 0.0); g.iter_mut().for_each(|z| *z = 0.0); let mut dev = 0.0;
        for r in 0..n { let vr = v[r]; if vr <= 0.0 { continue; }
            let (lo, hi) = (c.rp[r], c.rp[r + 1]); let idx = &c.ci[lo..hi]; let val = &c.cv[lo..hi];
            let mut e = 0.0; for j in 0..idx.len() { e += val[j] * beta[idx[j] as usize]; }
            let mu = expit(e); let yy = y[r];
            let mc = mu.clamp(1e-300, 1.0 - 1e-16);
            if yy > 0.0 { dev += vr * yy * (yy / mc).ln(); } if yy < 1.0 { dev += vr * (1.0 - yy) * ((1.0 - yy) / (1.0 - mc)).ln(); }
            let var = (mu * (1.0 - mu)).max(1e-300); let w = vr * var; let z = e + (yy - mu) / var;
            for j in 0..idx.len() { let a = idx[j] as usize; let wa = w * val[j]; g[a] += wa * z;
                for k in j..idx.len() { h[a * p + idx[k] as usize] += wa * val[k]; } } }
        dev *= 2.0;
        if dold.is_finite() && (dev - dold).abs() <= 1e-12 * (dold.abs() + 1.0) { return (beta, true); }
        dold = dev;
        for a in 0..p { for b in 0..a { h[a * p + b] = h[b * p + a]; } }
        let mut hh = h.clone(); let mut gg = g.clone();
        if !cholesky_solve(&mut hh, &mut gg, p) { for a in 0..p { h[a * p + a] += 1e-10 * (1.0 + h[a * p + a]); } hh = h.clone(); gg = g.clone(); if !cholesky_solve(&mut hh, &mut gg, p) { return (beta, false); } }
        let step = gg.iter().zip(&beta).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        let scale = gg.iter().map(|a| a.abs()).fold(0.0, f64::max);
        beta = gg;
        if step <= 1e-10 * (1.0 + scale) { return (beta, true); }
    }
    (beta, false)
}

/// Hessian X'WX at beta (weights v * mu * (1 - mu)), returned as its Cholesky factor (lower, row-major).
fn hessian_chol(c: &Class, v: &[f64], beta: &[f64]) -> Option<Vec<f64>> {
    let p = c.p; let mut h = vec![0.0; p * p];
    for r in 0..v.len() { let vr = v[r]; if vr <= 0.0 { continue; }
        let (lo, hi) = (c.rp[r], c.rp[r + 1]); let idx = &c.ci[lo..hi]; let val = &c.cv[lo..hi];
        let mut e = 0.0; for j in 0..idx.len() { e += val[j] * beta[idx[j] as usize]; }
        let mu = expit(e); let w = vr * (mu * (1.0 - mu)).max(1e-300);
        for j in 0..idx.len() { let a = idx[j] as usize; let wa = w * val[j]; for k in j..idx.len() { h[a * p + idx[k] as usize] += wa * val[k]; } } }
    for a in 0..p { for b in 0..a { h[a * p + b] = h[b * p + a]; } }
    // singular designs (e.g. BMI and BMI squared in a 2-unit bin): same ridge as irls(), increased if needed
    for ridge in [0.0, 1e-10, 1e-8, 1e-6] {
        let mut hh = h.clone(); for a in 0..p { hh[a * p + a] += ridge * (1.0 + h[a * p + a]); }
        let mut g = vec![0.0; p]; if cholesky_solve(&mut hh, &mut g, p) { return Some(hh); }
    }
    None
}

fn chol_apply(l: &[f64], g: &mut [f64], p: usize) {
    for i in 0..p { let mut s = g[i]; for k in 0..i { s -= l[i * p + k] * g[k]; } g[i] = s / l[i * p + i]; }
    for i in (0..p).rev() { let mut s = g[i]; for k in i + 1..p { s -= l[k * p + i] * g[k]; } g[i] = s / l[i * p + i]; }
}

/// Chord (fixed-Hessian) Newton: beta <- beta + H0^-1 X'(v (y - mu)). Converges to the same MLE as IRLS; each
/// iteration costs one sparse pass for the score. Falls back to full IRLS if it has not converged in 40 iterations.
fn chord(c: &Class, y: &[f64], v: &[f64], start: &[f64], l0: &[f64]) -> (Vec<f64>, bool) {
    let p = c.p; let mut beta = start.to_vec();
    for _it in 0..40 {
        let mut g = vec![0.0; p];
        for r in 0..v.len() { let vr = v[r]; if vr <= 0.0 { continue; }
            let (lo, hi) = (c.rp[r], c.rp[r + 1]); let idx = &c.ci[lo..hi]; let val = &c.cv[lo..hi];
            let mut e = 0.0; for j in 0..idx.len() { e += val[j] * beta[idx[j] as usize]; }
            let res = vr * (y[r] - expit(e)); for j in 0..idx.len() { g[idx[j] as usize] += res * val[j]; } }
        chol_apply(l0, &mut g, p);
        N_IT.fetch_add(1, Relaxed);
        let step = g.iter().map(|a| a.abs()).fold(0.0, f64::max);
        for a in 0..p { beta[a] += g[a]; }
        let scale = beta.iter().map(|a| a.abs()).fold(0.0, f64::max);
        if step <= 1e-8 * (1.0 + scale) { return (beta, true); }
    }
    irls(c, y, v, Some(&beta))
}

/// Chord method for B problems that share the design matrix (B bootstrap draws): each pass reads every row once and
/// updates all unconverged draws. Same iterates as chord() run separately. Falls back to IRLS per draw.
fn chord_batch(c: &Class, ys: &[Vec<f64>], vs: &[Vec<f64>], start: &[f64], l0: &[f64]) -> Vec<(Vec<f64>, bool)> {
    let p = c.p; let nb = ys.len(); let nn = c.ncol.len();
    let mut beta: Vec<f64> = (0..nb).flat_map(|_| start.iter().cloned()).collect();
    let mut active: Vec<usize> = (0..nb).collect(); let mut done = vec![false; nb];
    let mut g = vec![0.0; nb * p];
    for _it in 0..40 {
        if active.is_empty() { break; }
        for &b in &active { g[b * p..(b + 1) * p].iter_mut().for_each(|z| *z = 0.0); }
        for r in 0..c.npat {
            let bi = &c.bci[c.brp[r] as usize..c.brp[r + 1] as usize]; let nv = &c.nval[r * nn..(r + 1) * nn];
            for &b in &active {
                let vr = vs[b][r]; if vr <= 0.0 { continue; }
                let be = &beta[b * p..(b + 1) * p];
                let mut e = 0.0; for &a in bi { e += be[a as usize]; } for j in 0..nn { e += nv[j] * be[c.ncol[j]]; }
                let res = vr * (ys[b][r] - expit(e)); let gb = &mut g[b * p..(b + 1) * p];
                for &a in bi { gb[a as usize] += res; } for j in 0..nn { gb[c.ncol[j]] += res * nv[j]; }
            }
        }
        N_IT.fetch_add(active.len() as u64, Relaxed);
        let mut still = vec![];
        for &b in &active {
            let gb = &mut g[b * p..(b + 1) * p]; chol_apply(l0, gb, p);
            let step = gb.iter().map(|a| a.abs()).fold(0.0, f64::max);
            for a in 0..p { beta[b * p + a] += gb[a]; }
            let scale = beta[b * p..(b + 1) * p].iter().map(|a| a.abs()).fold(0.0, f64::max);
            if step <= 1e-8 * (1.0 + scale) { done[b] = true; } else { still.push(b); }
        }
        active = still;
    }
    (0..nb).map(|b| { let bb = beta[b * p..(b + 1) * p].to_vec(); if done[b] { (bb, true) } else { irls(c, &ys[b], &vs[b], Some(&bb)) } }).collect()
}

/// Weighted quantile by selection (expected O(n)): smallest x with cumulative weight share >= q, as wquant().
fn wquant_select(x: &[f64], w: &[f64], q: f64) -> f64 {
    let tot: f64 = w.iter().sum(); let mut items: Vec<(f64, f64)> = x.iter().cloned().zip(w.iter().cloned()).collect();
    let mut acc = 0.0;
    loop {
        if items.len() <= 64 {
            items.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            for &(v, ww) in &items { acc += ww; if acc / tot >= q { return v; } }
            return items.last().unwrap().0;
        }
        let m = items.len(); let (a, b, cc) = (items[0].0, items[m / 2].0, items[m - 1].0);
        let piv = if (a <= b) == (b <= cc) { b } else if (b <= a) == (a <= cc) { a } else { cc };
        let (mut lo, mut eq, mut hi) = (vec![], 0.0, vec![]);
        for &it in &items { if it.0 < piv { lo.push(it); } else if it.0 > piv { hi.push(it); } else { eq += it.1; } }
        let wl: f64 = lo.iter().map(|z| z.1).sum();
        if (acc + wl) / tot >= q { items = lo; }
        else if (acc + wl + eq) / tot >= q { return piv; }
        else { acc += wl + eq; items = hi; }
    }
}

/// Weights for B draws at once (chord refits share passes over the design matrix).
fn fit_weights_batch(d: &Data, ts: &[Vec<f64>], start: &Vec<(Vec<f64>, Vec<f64>)>, h0: &Vec<(Vec<f64>, Vec<f64>)>) -> Vec<(Vec<f64>, Vec<f64>)> {
    let nb = ts.len(); let n = d.risk.len();
    let mut out: Vec<(Vec<f64>, Vec<f64>)> = (0..nb).map(|_| (vec![0.0; n], vec![0.0; n])).collect();
    for (k, c) in d.classes.iter().enumerate() {
        let t0 = std::time::Instant::now();
        let (mut y1, mut v1, mut y2, mut v2) = (vec![], vec![], vec![], vec![]);
        for b in 0..nb {
            let mut cnt = vec![[0.0f64; 3]; c.npat];
            for &i in &c.cells { cnt[d.pid[i]][d.course[i] as usize] += ts[b][i]; }
            let tot: Vec<f64> = cnt.iter().map(|z| z[0] + z[1] + z[2]).collect(); let non: Vec<f64> = cnt.iter().map(|z| z[1] + z[2]).collect();
            y1.push(cnt.iter().zip(&tot).map(|(z, &s)| if s > 0.0 { z[0] / s } else { 0.0 }).collect::<Vec<f64>>());
            y2.push(cnt.iter().zip(&non).map(|(z, &s)| if s > 0.0 { z[2] / s } else { 0.0 }).collect::<Vec<f64>>());
            v1.push(tot); v2.push(non);
        }
        tick(&T_COUNT, t0); let t1 = std::time::Instant::now();
        let f1 = chord_batch(c, &y1, &v1, &start[k].0, &h0[k].0); let f2 = chord_batch(c, &y2, &v2, &start[k].1, &h0[k].1);
        tick(&T_GLM, t1);
        for b in 0..nb {
            let t2 = std::time::Instant::now();
            let (b1, b2) = (&f1[b].0, &f2[b].0); let nn = c.ncol.len();
            let lin = |be: &[f64], r: usize| -> f64 { let mut e = 0.0; for &a in &c.bci[c.brp[r] as usize..c.brp[r + 1] as usize] { e += be[a as usize]; }
                for j in 0..nn { e += c.nval[r * nn + j] * be[c.ncol[j]]; } e };
            let pi: Vec<f64> = (0..c.npat).map(|r| expit(lin(b1, r)).clamp(1e-4, 1.0 - 1e-4)).collect();
            let po: Vec<f64> = (0..c.npat).map(|r| (1.0 - pi[r]) * expit(lin(b2, r)).clamp(1e-6, 1.0 - 1e-6)).collect();
            let ob = &mut out[b]; let (wa, wb) = (&mut ob.0, &mut ob.1); let t = &ts[b];
            let (mut xa, mut ta, mut xb, mut tb) = (vec![], vec![], vec![], vec![]);
            for &i in &c.cells { let r = d.pid[i]; match d.course[i] {
                0 => { wa[i] = 1.0 + po[r] / pi[r]; wb[i] = 0.0; xa.push(wa[i]); ta.push(t[i]); }
                1 => { wa[i] = 1.0; wb[i] = 1.0 / (1.0 - pi[r]); xb.push(wb[i]); tb.push(t[i]); }
                _ => { wa[i] = 0.0; wb[i] = 1.0 / (1.0 - pi[r]); xb.push(wb[i]); tb.push(t[i]); } } }
            tick(&T_PRED, t2); let t3 = std::time::Instant::now();
            let ha = if xa.is_empty() { f64::INFINITY } else { wquant_select(&xa, &ta, 0.99) };
            let hb = if xb.is_empty() { f64::INFINITY } else { wquant_select(&xb, &tb, 0.99) };
            for &i in &c.cells { wa[i] = wa[i].min(ha); wb[i] = wb[i].min(hb); }
            tick(&T_QUANT, t3);
        }
    }
    out
}

// ---------- data ----------
struct Class { npat: usize, p: usize, x: Vec<f64>, cells: Vec<usize>, rp: Vec<usize>, ci: Vec<u16>, cv: Vec<f64>,
               brp: Vec<u32>, bci: Vec<u8>, ncol: Vec<usize>, nval: Vec<f64> }
struct Data { cls: Vec<usize>, pid: Vec<usize>, course: Vec<u8>, risk: Vec<f64>, live: Vec<f64>, ev: Vec<Vec<f64>>, ev_cm: Vec<f64>, den_live: Vec<bool>,
              classes: Vec<Class>, names: Vec<String>, outcomes: Vec<String> }

fn read_f64(path: &str) -> Vec<f64> { let b = fs::read(path).expect(path); b.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect() }

fn load(dir: &str) -> Data {
    let meta: Value = serde_json::from_str(&fs::read_to_string(format!("{dir}/meta.json")).unwrap()).unwrap();
    let ncells = meta["ncells"].as_u64().unwrap() as usize;
    let outs: Vec<(String, bool)> = meta["outcomes"].as_array().unwrap().iter().map(|o| (o["name"].as_str().unwrap().to_string(), o["den"].as_str().unwrap() == "live")).collect();
    let col = read_f64(&format!("{dir}/cells.bin")); assert_eq!(col.len(), ncells * (5 + outs.len()));
    let get = |j: usize| col[j * ncells..(j + 1) * ncells].to_vec();
    let cls: Vec<usize> = get(0).iter().map(|&v| v as usize).collect(); let pid: Vec<usize> = get(1).iter().map(|&v| v as usize).collect();
    let course: Vec<u8> = get(2).iter().map(|&v| v as u8).collect();
    let ev: Vec<Vec<f64>> = (0..outs.len()).map(|o| get(5 + o)).collect();
    let mut classes = vec![]; let mut names = vec![];
    for (k, cm) in meta["classes"].as_array().unwrap().iter().enumerate() {
        let npat = cm["npat"].as_u64().unwrap() as usize; let p = cm["p"].as_u64().unwrap() as usize;
        let x = read_f64(&format!("{dir}/X_{k}.bin")); assert_eq!(x.len(), npat * p);
        let cells: Vec<usize> = (0..ncells).filter(|&c| cls[c] == k).collect();
        let (mut rp, mut ci, mut cv) = (vec![0usize], vec![], vec![]);
        for r in 0..npat { for a in 0..p { let v = x[r * p + a]; if v != 0.0 { ci.push(a as u16); cv.push(v); } } rp.push(ci.len()); }
        // compact form: binary columns (all values 0 or 1) as index lists, other columns dense
        let binary: Vec<bool> = (0..p).map(|a| (0..npat).all(|r| { let v = x[r * p + a]; v == 0.0 || v == 1.0 })).collect();
        let ncol: Vec<usize> = (0..p).filter(|&a| !binary[a]).collect();
        let (mut brp, mut bci, mut nval) = (vec![0u32], vec![], Vec::with_capacity(npat * ncol.len()));
        for r in 0..npat { for a in 0..p { if binary[a] && x[r * p + a] == 1.0 { bci.push(a as u8); } } brp.push(bci.len() as u32);
            for &a in &ncol { nval.push(x[r * p + a]); } }
        assert!(p < 256);
        classes.push(Class { npat, p, x, cells, rp, ci, cv, brp, bci, ncol, nval }); names.push(cm["name"].as_str().unwrap().to_string());
    }
    let no = ev.len(); let mut ev_cm = vec![0.0; ncells * no];
    for o in 0..no { for c in 0..ncells { ev_cm[c * no + o] = ev[o][c]; } }
    Data { cls, pid, course, risk: get(3), live: get(4), ev, ev_cm, den_live: outs.iter().map(|o| o.1).collect(), classes, names, outcomes: outs.iter().map(|o| o.0.clone()).collect() }
}

fn wquant(x: &[f64], w: &[f64], q: f64) -> f64 {
    let mut o: Vec<usize> = (0..x.len()).collect(); o.sort_by(|&a, &b| x[a].partial_cmp(&x[b]).unwrap().then(a.cmp(&b)));
    let tot: f64 = w.iter().sum(); let mut cw = 0.0;
    for &i in &o { cw += w[i]; if cw / tot >= q { return x[i]; } } x[*o.last().unwrap()]
}

struct Fit { wa: Vec<f64>, wb: Vec<f64>, params: Vec<(Vec<f64>, Vec<f64>)>, trunc: Vec<(f64, f64)>, conv: Vec<bool> }

/// Fit both treatment models in every class on cell totals t, return per-cell weights (indexed like cells).
fn fit_weights(d: &Data, t: &[f64], start: Option<&Vec<(Vec<f64>, Vec<f64>)>>) -> Fit { fit_weights_h(d, t, start, None) }

/// h0: per class Cholesky factors of the two point-fit Hessians; when given, refits use the chord method.
fn fit_weights_h(d: &Data, t: &[f64], start: Option<&Vec<(Vec<f64>, Vec<f64>)>>, h0: Option<&Vec<(Vec<f64>, Vec<f64>)>>) -> Fit {
    let n = t.len(); let mut wa = vec![0.0; n]; let mut wb = vec![0.0; n]; let mut params = vec![]; let mut trunc = vec![]; let mut conv = vec![];
    for (k, c) in d.classes.iter().enumerate() {
        let t0 = std::time::Instant::now();
        let mut cnt = vec![[0.0f64; 3]; c.npat];
        for &i in &c.cells { cnt[d.pid[i]][d.course[i] as usize] += t[i]; }
        let tot: Vec<f64> = cnt.iter().map(|z| z[0] + z[1] + z[2]).collect();
        let y1: Vec<f64> = cnt.iter().zip(&tot).map(|(z, &s)| if s > 0.0 { z[0] / s } else { 0.0 }).collect();
        let non: Vec<f64> = cnt.iter().map(|z| z[1] + z[2]).collect();
        let y2: Vec<f64> = cnt.iter().zip(&non).map(|(z, &s)| if s > 0.0 { z[2] / s } else { 0.0 }).collect();
        tick(&T_COUNT, t0); let t1 = std::time::Instant::now();
        let ((b1, c1), (b2, c2)) = match (h0, start) {
            (Some(h), Some(s)) => (chord(c, &y1, &tot, &s[k].0, &h[k].0), chord(c, &y2, &non, &s[k].1, &h[k].1)),
            _ => (irls(c, &y1, &tot, start.map(|s| s[k].0.as_slice())), irls(c, &y2, &non, start.map(|s| s[k].1.as_slice()))) };
        tick(&T_GLM, t1); let t2 = std::time::Instant::now();
        let mut pi = vec![0.0; c.npat]; let mut po = vec![0.0; c.npat];
        for r in 0..c.npat {
            let (mut e1, mut e2) = (0.0, 0.0); for j in c.rp[r]..c.rp[r + 1] { let a = c.ci[j] as usize; e1 += c.cv[j] * b1[a]; e2 += c.cv[j] * b2[a]; }
            pi[r] = expit(e1).clamp(1e-4, 1.0 - 1e-4); po[r] = (1.0 - pi[r]) * expit(e2).clamp(1e-6, 1.0 - 1e-6); }
        let (mut xa, mut ta, mut xb, mut tb) = (vec![], vec![], vec![], vec![]);
        for &i in &c.cells { let r = d.pid[i]; match d.course[i] {
            0 => { wa[i] = 1.0 + po[r] / pi[r]; wb[i] = 0.0; xa.push(wa[i]); ta.push(t[i]); }
            1 => { wa[i] = 1.0; wb[i] = 1.0 / (1.0 - pi[r]); xb.push(wb[i]); tb.push(t[i]); }
            _ => { wa[i] = 0.0; wb[i] = 1.0 / (1.0 - pi[r]); xb.push(wb[i]); tb.push(t[i]); } } }
        tick(&T_PRED, t2); let t3 = std::time::Instant::now();
        let ha = if xa.is_empty() { f64::INFINITY } else { wquant(&xa, &ta, 0.99) }; let hb = if xb.is_empty() { f64::INFINITY } else { wquant(&xb, &tb, 0.99) };
        for &i in &c.cells { wa[i] = wa[i].min(ha); wb[i] = wb[i].min(hb); }
        tick(&T_QUANT, t3);
        params.push((b1, b2)); trunc.push((ha, hb)); conv.push(c1 && c2);
    }
    Fit { wa, wb, params, trunc, conv }
}

/// Risks per outcome and class in one pass over cells. Denominators are shared by all outcomes with the same
/// denominator type, and cells without events skip the event draw. rng None = point estimate (observed events),
/// Some = Beta-drawn event fractions. rec (dump mode) receives the drawn events per outcome.
fn risks(d: &Data, t: &[f64], f: &Fit, mut rng: Option<&mut Rng>, mut rec: Option<&mut Vec<Vec<f64>>>) -> Vec<Vec<[f64; 2]>> {
    let k = d.classes.len(); let no = d.outcomes.len(); let n = t.len();
    let freq = freq_boot();
    let mut num = vec![[0.0f64; 2]; k * no]; let mut den = vec![[0.0f64; 4]; k];
    if let Some(r) = rec.as_deref_mut() { *r = vec![vec![0.0; n]; no]; }
    for i in 0..n {
        let c = d.cls[i]; let (wa, wb) = (f.wa[i], f.wb[i]);
        let dr = t[i];
        let dl = if d.risk[i] <= 0.0 { 0.0 } else { match (freq, rng.as_deref_mut()) {
            (true, Some(r)) => r.binomial(t[i], d.live[i] / d.risk[i]), _ => t[i] * d.live[i] / d.risk[i] } };
        let dc = &mut den[c]; dc[0] += wa * dr; dc[1] += wb * dr; dc[2] += wa * dl; dc[3] += wb * dl;
        let evs = &d.ev_cm[i * no..(i + 1) * no];
        for o in 0..no {
            let e = evs[o]; if e <= 0.0 { continue; }
            let (nn, dn) = if d.den_live[o] { (d.live[i], dl) } else { (d.risk[i], dr) };
            let ev = match rng.as_deref_mut() { None => if nn > 0.0 { e * dn / nn } else { 0.0 },
                Some(r) => if freq { r.binomial(dn, (e / nn).min(1.0)) } else if e >= nn { dn } else { dn * r.beta(e, nn - e) } };
            let q = &mut num[c * no + o]; q[0] += wa * ev; q[1] += wb * ev;
            if let Some(r) = rec.as_deref_mut() { r[o][i] = ev; }
        }
    }
    (0..no).map(|o| (0..k).map(|c| { let (ta, tb) = if d.den_live[o] { (den[c][2], den[c][3]) } else { (den[c][0], den[c][1]) };
        [if ta > 0.0 { num[c * no + o][0] / ta } else { f64::NAN }, if tb > 0.0 { num[c * no + o][1] / tb } else { f64::NAN }] }).collect()).collect()
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let dir = &a[2]; let d = load(dir);
    let point = fit_weights(&d, &d.risk, None);
    let point_h = || -> Vec<(Vec<f64>, Vec<f64>)> { d.classes.iter().enumerate().map(|(k, c)| {
        let mut cnt = vec![[0.0f64; 3]; c.npat]; for &i in &c.cells { cnt[d.pid[i]][d.course[i] as usize] += d.risk[i]; }
        let tot: Vec<f64> = cnt.iter().map(|z| z[0] + z[1] + z[2]).collect(); let non: Vec<f64> = cnt.iter().map(|z| z[1] + z[2]).collect();
        (hessian_chol(c, &tot, &point.params[k].0).expect("point hessian 1"), hessian_chol(c, &non, &point.params[k].1).expect("point hessian 2")) }).collect() };
    if a[1] == "point" || a[1] == "glmtest" {
        let r = risks(&d, &d.risk, &point, None, None);
        let js = json!({
            "classes": d.names, "outcomes": d.outcomes,
            "params": point.params.iter().map(|(b1, b2)| json!([b1, b2])).collect::<Vec<_>>(),
            "trunc": point.trunc.iter().map(|(x, y)| json!([x, y])).collect::<Vec<_>>(), "converged": point.conv,
            "risks": r.iter().map(|ro| ro.iter().map(|v| json!([v[0], v[1]])).collect::<Vec<_>>()).collect::<Vec<_>>(),
        });
        fs::write(format!("{dir}/point.json"), js.to_string()).unwrap();
        let mut buf: Vec<u8> = Vec::with_capacity(point.wa.len() * 16);
        for v in point.wa.iter().chain(point.wb.iter()) { buf.extend_from_slice(&v.to_le_bytes()); }
        fs::write(format!("{dir}/point_weights.bin"), buf).unwrap(); return;
    }
    if a[1] == "bench" {
        let nd: usize = a[3].parse().unwrap(); let th: usize = a[4].parse().unwrap();
        rayon::ThreadPoolBuilder::new().num_threads(th).build_global().unwrap();
        let h0 = point_h(); let t0 = std::time::Instant::now();
        let bsz: usize = std::env::var("CCW_BATCH").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
        let jobs: Vec<Vec<usize>> = (0..nd).collect::<Vec<_>>().chunks(bsz).map(|c| c.to_vec()).collect();
        let parts: Vec<(f64, f64, f64)> = jobs.into_par_iter().map(|js| {
            let s0 = std::time::Instant::now();
            let mut rngs: Vec<Rng> = js.iter().map(|&j| Rng::new(99 + j as u64)).collect();
            let ts: Vec<Vec<f64>> = rngs.iter_mut().map(|rng| d.risk.iter().map(|&n| total_draw(rng, n, freq_boot())).collect()).collect(); let s1 = s0.elapsed().as_secs_f64();
            let ws = fit_weights_batch(&d, &ts, &point.params, &h0); let s2 = s0.elapsed().as_secs_f64();
            for b in 0..js.len() { let f = Fit { wa: ws[b].0.clone(), wb: ws[b].1.clone(), params: vec![], trunc: vec![], conv: vec![] }; let _ = risks(&d, &ts[b], &f, Some(&mut rngs[b]), None); }
            let s3 = s0.elapsed().as_secs_f64(); (s1, s2 - s1, s3 - s2) }).collect();
        let (g, fw, rk) = parts.iter().fold((0.0, 0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2));
        println!("{{\"draws\": {nd}, \"threads\": {th}, \"wall_s\": {:.2}, \"per_draw_cpu_s\": {{\"gamma_totals\": {:.4}, \"fit_weights\": {:.4}, \"risks_beta\": {:.4}}}}}",
                 t0.elapsed().as_secs_f64(), g / nd as f64, fw / nd as f64, rk / nd as f64);
        let f = |a: &AtomicU64| a.load(Relaxed) as f64 / 1e9 / nd as f64;
        println!("fit_weights split per draw (cpu s): count {:.4}, glm {:.4}, predict+weights {:.4}, quantile {:.4}; chord iterations per fit {:.1}",
                 f(&T_COUNT), f(&T_GLM), f(&T_PRED), f(&T_QUANT), N_IT.load(Relaxed) as f64 / nd as f64 / (2.0 * d.classes.len() as f64)); return;
    }
    if a[1] == "dump" {
        // one draw: write drawn totals t, drawn events per outcome, and the rust risks, for an exact check in python
        let seed: u64 = a[3].parse().unwrap(); let j: u64 = a[4].parse().unwrap();
        let mut rng = Rng::new(seed.wrapping_mul(1_000_003).wrapping_add(j));
        let t: Vec<f64> = d.risk.iter().map(|&n| total_draw(&mut rng, n, freq_boot())).collect();
        let h0 = point_h(); let f = fit_weights_h(&d, &t, Some(&point.params), Some(&h0)); let mut rec: Vec<Vec<f64>> = vec![];
        let r = risks(&d, &t, &f, Some(&mut rng), Some(&mut rec));
        let mut buf: Vec<u8> = vec![]; for v in &t { buf.extend_from_slice(&v.to_le_bytes()); }
        for ro in &rec { for v in ro { buf.extend_from_slice(&v.to_le_bytes()); } }
        fs::write(format!("{dir}/dump.bin"), buf).unwrap();
        let js = json!({"risks": r.iter().map(|ro| ro.iter().map(|v| json!([v[0], v[1]])).collect::<Vec<_>>()).collect::<Vec<_>>(),
                        "params": f.params.iter().map(|(b1, b2)| json!([b1, b2])).collect::<Vec<_>>()});
        fs::write(format!("{dir}/dump.json"), js.to_string()).unwrap(); return;
    }
    let nd: usize = a[3].parse().unwrap(); let seed: u64 = a[4].parse().unwrap(); let th: usize = a[5].parse().unwrap();
    rayon::ThreadPoolBuilder::new().num_threads(th).build_global().unwrap();
    let h0 = point_h(); let bsz: usize = std::env::var("CCW_BATCH").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
    let jobs: Vec<Vec<usize>> = (0..nd).collect::<Vec<_>>().chunks(bsz).map(|c| c.to_vec()).collect();
    let res: Vec<Vec<Vec<[f64; 2]>>> = jobs.into_par_iter().flat_map_iter(|js| {
        let mut rngs: Vec<Rng> = js.iter().map(|&j| Rng::new(seed.wrapping_mul(1_000_003).wrapping_add(j as u64))).collect();
        let ts: Vec<Vec<f64>> = rngs.iter_mut().map(|rng| d.risk.iter().map(|&n| total_draw(rng, n, freq_boot())).collect()).collect();
        let ws = fit_weights_batch(&d, &ts, &point.params, &h0);
        let out: Vec<Vec<Vec<[f64; 2]>>> = (0..js.len()).map(|b| { let f = Fit { wa: ws[b].0.clone(), wb: ws[b].1.clone(), params: vec![], trunc: vec![], conv: vec![] };
            risks(&d, &ts[b], &f, Some(&mut rngs[b]), None) }).collect();
        out.into_iter()
    }).collect();
    let mut buf: Vec<u8> = Vec::with_capacity(nd * d.outcomes.len() * d.classes.len() * 16);
    for r in &res { for ro in r { for v in ro { buf.extend_from_slice(&v[0].to_le_bytes()); buf.extend_from_slice(&v[1].to_le_bytes()); } } }
    fs::write(format!("{dir}/draws.bin"), buf).unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;
    fn moments(f: &mut dyn FnMut() -> f64, n: usize) -> (f64, f64) {
        let v: Vec<f64> = (0..n).map(|_| f()).collect(); let m = v.iter().sum::<f64>() / n as f64;
        (m, v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (n - 1) as f64)
    }
    #[test]
    fn poisson_binomial_moments() {
        let mut r = Rng::new(11);
        for &l in &[0.3, 4.0, 12.0, 250.0, 40000.0] {
            let (m, v) = moments(&mut || r.poisson(l), 200_000);
            assert!((m - l).abs() < 5.0 * (l / 200_000f64).sqrt(), "poisson mean l={l} m={m}");
            assert!((v / l - 1.0).abs() < 0.03, "poisson var l={l} v={v}");
        }
        for &(n, p) in &[(10.0, 0.3), (65.0, 0.02), (900.0, 0.004), (50000.0, 0.6), (300.0, 0.97)] {
            let (m, v) = moments(&mut || r.binomial(n, p), 200_000); let ev = n * p * (1.0 - p);
            assert!((m - n * p).abs() < 5.0 * (ev / 200_000f64).sqrt() + 1e-9, "binom mean n={n} p={p} m={m}");
            assert!((v / ev - 1.0).abs() < 0.03, "binom var n={n} p={p} v={v}");
        }
        assert!((ln_gamma(10.0) - 362880f64.ln()).abs() < 1e-9);
    }

    #[test]
    fn gamma_moments() {
        let mut r = Rng::new(1);
        for &a in &[0.3, 1.0, 2.5, 40.0, 5000.0] {
            let (m, v) = moments(&mut || r.gamma(a), 400_000);
            assert!((m - a).abs() < 5.0 * (a / 400_000f64).sqrt(), "gamma mean a={a} m={m}");
            assert!((v / a - 1.0).abs() < 0.03, "gamma var a={a} v={v}");
        }
    }
    #[test]
    fn beta_moments() {
        let mut r = Rng::new(2);
        for &(a, b) in &[(1.0, 1.0), (2.0, 5.0), (0.5, 30.0), (300.0, 9000.0)] {
            let (m, v) = moments(&mut || r.beta(a, b), 400_000);
            let em = a / (a + b); let ev = a * b / ((a + b) * (a + b) * (a + b + 1.0));
            assert!((m - em).abs() < 5.0 * (ev / 400_000f64).sqrt(), "beta mean {a},{b}");
            assert!((v / ev - 1.0).abs() < 0.03, "beta var {a},{b}");
        }
    }
    #[test]
    fn wquant_select_matches_sort() {
        let mut r = Rng::new(5);
        for n in [10usize, 1000, 50000] {
            let x: Vec<f64> = (0..n).map(|i| ((r.unif() * 50.0).floor() + if i % 7 == 0 { 0.5 } else { 0.0 })).collect();
            let w: Vec<f64> = (0..n).map(|_| r.gamma(2.0)).collect();
            for q in [0.01, 0.5, 0.99] { assert_eq!(wquant(&x, &w, q), wquant_select(&x, &w, q)); }
        }
    }
    #[test]
    fn irls_matches_closed_form() {
        // intercept-only model: beta = logit(sum(v*y)/sum(v))
        let c = Class { npat: 4, p: 1, x: vec![1.0; 4], cells: vec![], rp: vec![0, 1, 2, 3, 4], ci: vec![0; 4], cv: vec![1.0; 4], brp: vec![0, 1, 2, 3, 4], bci: vec![0; 4], ncol: vec![], nval: vec![] };
        let y = vec![0.1, 0.3, 0.0, 1.0]; let v = vec![10.0, 5.0, 2.0, 3.0];
        let (b, ok) = irls(&c, &y, &v, None); let p: f64 = (1.0 + 1.5 + 0.0 + 3.0) / 20.0;
        assert!(ok && (b[0] - (p / (1.0 - p)).ln()).abs() < 1e-10);
    }
}
