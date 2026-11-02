//! Pre-registered locked gate runner (`G1..G9` + `I1..I3`).
//!
//! Usage: `alpha11-gates <core> [dev|dev30|heldout|ablate|diag] [--open-vault]`

use std::env;
use std::fs;
use std::path::PathBuf;
use std::thread;

use a11_core::causal_core::{CausalAblation, CausalCore, CausalParams};
use a11_core::{Alpha11Organism, OrganismConfig};
use a11_medium::OrganismEngine;
use a11_observe::gates::{
    BrightArms, GateTicks, benchmark_dense_4096, evaluate_bright_gates, evaluate_invariants,
    legacy_reversal_rows, run_bright_seed,
};

fn run_family<O, M>(make: M, seeds: &[u64]) -> (BrightArms, (f64, f64, f64, usize))
where
    O: OrganismEngine + Clone,
    M: Fn(u64) -> O + Sync,
{
    let ticks = GateTicks::default();
    let per_seed: Vec<BrightArms> = thread::scope(|sc| {
        let handles: Vec<_> = seeds
            .iter()
            .map(|&s| {
                let mk = &make;
                sc.spawn(move || {
                    let mut a = BrightArms::default();
                    run_bright_seed(mk, s, ticks, &mut a);
                    a
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("seed thread"))
            .collect()
    });
    let mut all = BrightArms::default();
    for a in per_seed {
        all.natural.extend(a.natural);
        all.frozen.extend(a.frozen);
        all.yoked.extend(a.yoked);
        all.shuffled.extend(a.shuffled);
        all.xfer_trained.extend(a.xfer_trained);
        all.xfer_newborn.extend(a.xfer_newborn);
        all.xfer_lesion.extend(a.xfer_lesion);
        all.rev_post_natural.extend(a.rev_post_natural);
        all.rev_post_frozen.extend(a.rev_post_frozen);
        all.reacq_first_post.extend(a.reacq_first_post);
        all.reacq_first_life.extend(a.reacq_first_life);
        all.rev_post_frozen_clone.extend(a.rev_post_frozen_clone);
        all.reacq_gain_post.extend(a.reacq_gain_post);
        all.acq_gain_life.extend(a.acq_gain_life);
        all.abab_return_natural.extend(a.abab_return_natural);
        all.abab_return_newborn.extend(a.abab_return_newborn);
        all.torture_natural.extend(a.torture_natural);
        all.torture_frozen.extend(a.torture_frozen);
        all.agency_natural.extend(a.agency_natural);
        all.agency_shuffled.extend(a.agency_shuffled);
        all.agency_exogenous.extend(a.agency_exogenous);
        all.atp_natural.extend(a.atp_natural);
        all.atp_frozen.extend(a.atp_frozen);
        all.famine_mitoses.extend(a.famine_mitoses);
        all.torture_distractor_share
            .extend(a.torture_distractor_share);
    }
    let dense = benchmark_dense_4096(&make, 1_000);
    (all, dense)
}

fn fmt_vec(v: &[f64]) -> String {
    v.iter()
        .map(|x| format!("{x:+.1}"))
        .collect::<Vec<_>>()
        .join(" ")
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let core = args.get(1).map_or("causal", String::as_str).to_string();
    let split = args.get(2).map_or("dev30", String::as_str).to_string();
    let open_vault = args.iter().any(|a| a == "--open-vault");

    if split == "diag" {
        diag(&core);
        return;
    }
    if split == "sweep" {
        sweep();
        return;
    }
    if split == "ablate" {
        ablate();
        return;
    }
    let seeds: Vec<u64> = match split.as_str() {
        "dev" => (0..10).collect(),
        "dev30" => (0..30).collect(),
        "heldout" => {
            if !open_vault {
                eprintln!("Refusing: the held-out vault needs --open-vault (single use per core).");
                std::process::exit(2);
            }
            let ws_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
            if !ws_root.join(".vault_authorized").exists() {
                eprintln!(
                    "Refusing: .vault_authorized does not exist in workspace root (requires explicit user approval)."
                );
                std::process::exit(2);
            }
            let dir = ws_root.join("observations/heldout_vault");
            let _ = fs::create_dir_all(&dir);
            let marker = dir.join(format!("{core}.opened"));
            if marker.exists() {
                eprintln!(
                    "Refusing: held-out vault already opened for core {core} ({}).",
                    marker.display()
                );
                std::process::exit(3);
            }
            let _ = fs::write(&marker, b"opened\n");
            (200..212).collect()
        }
        other => {
            eprintln!("Unknown split {other}");
            std::process::exit(2);
        }
    };

    let (arms, dense) = match core.as_str() {
        "causal" => run_family(CausalCore::new, &seeds),
        "causal-no-cs" => run_family(
            |s| {
                CausalCore::with_ablation(
                    s,
                    CausalAblation {
                        no_center_surround: true,
                        ..Default::default()
                    },
                )
            },
            &seeds,
        ),
        "causal-no-td" => run_family(
            |s| {
                CausalCore::with_ablation(
                    s,
                    CausalAblation {
                        no_td_critic: true,
                        ..Default::default()
                    },
                )
            },
            &seeds,
        ),
        "alpha11" => run_family(
            |s| Alpha11Organism::new(s, OrganismConfig::default()),
            &seeds,
        ),
        other => {
            eprintln!("Unknown core {other}");
            std::process::exit(2);
        }
    };

    println!(
        "== Locked Gate Suite (G1..G9 + I1..I3) | core={core} | split={split} | n={} ==",
        seeds.len()
    );
    println!("natural      {}", fmt_vec(&arms.natural));
    println!("frozen       {}", fmt_vec(&arms.frozen));
    println!("yoked        {}", fmt_vec(&arms.yoked));
    println!("shuffled     {}", fmt_vec(&arms.shuffled));
    println!("xfer_trained {}", fmt_vec(&arms.xfer_trained));
    println!("xfer_newborn {}", fmt_vec(&arms.xfer_newborn));
    println!("xfer_lesion  {}", fmt_vec(&arms.xfer_lesion));
    println!("rev_post_nat {}", fmt_vec(&arms.rev_post_natural));
    println!("rev_post_cln {}", fmt_vec(&arms.rev_post_frozen_clone));
    println!("reacq_gain   {}", fmt_vec(&arms.reacq_gain_post));
    println!("acq_gain     {}", fmt_vec(&arms.acq_gain_life));
    println!("abab_ret_nat {}", fmt_vec(&arms.abab_return_natural));
    println!("abab_ret_nb  {}", fmt_vec(&arms.abab_return_newborn));
    println!("torture_nat  {}", fmt_vec(&arms.torture_natural));
    println!("torture_frz  {}", fmt_vec(&arms.torture_frozen));

    let rows = evaluate_bright_gates(&arms, 0.05);
    let invs = evaluate_invariants(&arms, dense);
    let mut all_pass = true;
    for r in &rows {
        all_pass &= r.pass;
        println!(
            "{:<50} A={:+8.2} B={:+8.2} diff={:+8.2} t={:+6.2} p={:.5} p_holm={:.5} wins={}/{} {}",
            r.name,
            r.stats.mean_a,
            r.stats.mean_b,
            r.stats.mean_diff,
            r.stats.t,
            r.stats.p_two_sided,
            r.p_holm,
            r.stats.wins,
            r.stats.n,
            if r.pass { "PASS" } else { "FAIL" }
        );
    }
    for inv in &invs {
        all_pass &= inv.pass;
        println!(
            "{:<50} {} | {}",
            inv.name,
            if inv.pass { "PASS" } else { "FAIL" },
            inv.detail
        );
    }
    let pass_cnt =
        rows.iter().filter(|r| r.pass).count() + invs.iter().filter(|i| i.pass).count();
    println!(
        "FULL SUITE VERDICT: {}/{} {}",
        pass_cnt,
        rows.len() + invs.len(),
        if all_pass { "PASS" } else { "FAIL" }
    );
    for (name, g) in legacy_reversal_rows(&arms) {
        println!(
            "[disclosure] {name:<46} diff={:+8.2} t={:+6.2} p={:.5} wins={}/{}",
            g.mean_diff, g.t, g.p_two_sided, g.wins, g.n
        );
    }
    let (used_128, used_4096) = measure_used_state_bytes(&core);
    let (vm_rss_kib, vm_hwm_kib) = proc_rss_hwm_kib();
    println!(
        "[memory] deterministic_used_state_bytes(128ch)={used_128} deterministic_used_state_bytes(4096ch)={used_4096} | os_vm_rss_kib={vm_rss_kib} os_vm_hwm_kib={vm_hwm_kib}"
    );
}

fn measure_used_state_bytes(core: &str) -> (usize, usize) {
    use a11_medium::FieldView;
    let buf128 = [0.0_f32; 128];
    let buf4096 = vec![0.0_f32; 4096];
    let view128 = FieldView {
        ambient_in: &buf128,
        metabolic_influx: 0.0,
        tissue_stress: 0.0,
        num_actuators: 3,
    };
    let view4096 = FieldView {
        ambient_in: &buf4096,
        metabolic_influx: 0.0,
        tissue_stress: 0.0,
        num_actuators: 3,
    };
    match core {
        "alpha11" => {
            let mut o1 = Alpha11Organism::new(1_000, OrganismConfig::default());
            let _ = o1.step(view128);
            let mut o2 = Alpha11Organism::new(1_000, OrganismConfig::default());
            let _ = o2.step(view4096);
            (o1.snapshot().used_state_bytes, o2.snapshot().used_state_bytes)
        }
        _ => {
            let mut o1 = CausalCore::new(1_000);
            let _ = o1.step(view128);
            let mut o2 = CausalCore::new(1_000);
            let _ = o2.step(view4096);
            (o1.snapshot().used_state_bytes, o2.snapshot().used_state_bytes)
        }
    }
}

fn proc_rss_hwm_kib() -> (u64, u64) {
    let Ok(txt) = fs::read_to_string("/proc/self/status") else {
        return (0, 0);
    };
    let mut rss = 0_u64;
    let mut hwm = 0_u64;
    for line in txt.lines() {
        if let Some(rest) = line.strip_prefix("VmRSS:") {
            rss = rest
                .split_whitespace()
                .next()
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);
        } else if let Some(rest) = line.strip_prefix("VmHWM:") {
            hwm = rest
                .split_whitespace()
                .next()
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);
        }
    }
    (rss, hwm)
}

fn diag(core: &str) {
    use a11_medium::WorldMedium;
    use a11_worlds::BrightRadiantWorld;
    for s in 0..3_u64 {
        let mut org = match core {
            "causal-no-td" => CausalCore::with_ablation(
                1_000 + s,
                CausalAblation {
                    no_td_critic: true,
                    ..Default::default()
                },
            ),
            _ => CausalCore::new(1_000 + s),
        };
        let mut w = BrightRadiantWorld::new(5_000 + s);
        let (mut inf, mut st) = (0.0_f64, 0.0_f64);
        for t in 1..=6_000_u64 {
            let p = org.step(w.current_view()).expect("step");
            w.apply_pulse(&p);
            let v = w.current_view();
            inf += f64::from(v.metabolic_influx);
            st += f64::from(v.tissue_stress);
            if t % 1_000 == 0 {
                let snap = org.snapshot();
                println!(
                    "seed {s} t={t:>5} influx={inf:7.2} stress={st:7.2} wall_hits={:>4} springs={:>3} ag={:8.4} td_abs={:.4} atp={:.2} pores={} pos=({:+.1},{:+.1})",
                    w.0.metrics().wall_impacts,
                    w.0.metrics().springs_depleted,
                    snap.mean_interventional_agency,
                    snap.mean_prediction_error,
                    snap.atp,
                    snap.sensor_pore_count,
                    w.0.metrics().position.0,
                    w.0.metrics().position.1
                );
                inf = 0.0;
                st = 0.0;
            }
        }
    }
}

fn sweep() {
    let seeds: Vec<u64> = (0..10).collect();
    let p = CausalParams::default();
    let (arms, _) = run_family(
        move |s| CausalCore::with_config(s, CausalAblation::default(), p),
        &seeds,
    );
    for r in evaluate_bright_gates(&arms, 0.05) {
        println!(
            "{:<50} t={:+6.2} {}",
            r.name,
            r.stats.t,
            if r.pass { "PASS" } else { "FAIL" }
        );
    }
}

fn ablate() {
    use a11_observe::gates::{holm_adjust, paired_gate};
    let seeds: Vec<u64> = (0..30).collect();
    let (intact, intact_dense) = run_family(CausalCore::new, &seeds);
    let variants: Vec<(&'static str, CausalAblation)> = vec![
        (
            "no_center_surround",
            CausalAblation {
                no_center_surround: true,
                ..Default::default()
            },
        ),
        (
            "no_td_critic",
            CausalAblation {
                no_td_critic: true,
                ..Default::default()
            },
        ),
        (
            "no_forward_agency",
            CausalAblation {
                no_forward_agency: true,
                ..Default::default()
            },
        ),
        (
            "no_pore_frontier",
            CausalAblation {
                no_pore_frontier: true,
                ..Default::default()
            },
        ),
        (
            "no_latent_context",
            CausalAblation {
                no_latent_context: true,
                ..Default::default()
            },
        ),
    ];
    let mut rows = Vec::new();
    let mut suite_summaries = Vec::new();
    {
        let r0 = evaluate_bright_gates(&intact, 0.05);
        let i0 = evaluate_invariants(&intact, intact_dense);
        let pass0 = r0.iter().filter(|r| r.pass).count() + i0.iter().filter(|i| i.pass).count();
        suite_summaries.push(("intact", pass0, r0, i0));
    }
    for (name, ab) in variants {
        let (arms, dense) = run_family(move |s| CausalCore::with_ablation(s, ab), &seeds);
        rows.push((name, "train net", paired_gate(&arms.natural, &intact.natural)));
        rows.push((
            name,
            "transfer net",
            paired_gate(&arms.xfer_trained, &intact.xfer_trained),
        ));
        rows.push((
            name,
            "post-reversal net",
            paired_gate(&arms.rev_post_natural, &intact.rev_post_natural),
        ));
        rows.push((
            name,
            "ABAB return net",
            paired_gate(&arms.abab_return_natural, &intact.abab_return_natural),
        ));
        rows.push((
            name,
            "torture net",
            paired_gate(&arms.torture_natural, &intact.torture_natural),
        ));
        let rv = evaluate_bright_gates(&arms, 0.05);
        let iv = evaluate_invariants(&arms, dense);
        let passv = rv.iter().filter(|r| r.pass).count() + iv.iter().filter(|i| i.pass).count();
        suite_summaries.push((name, passv, rv, iv));
    }
    let adj = holm_adjust(&rows.iter().map(|r| r.2.p_two_sided).collect::<Vec<_>>());
    println!("== Paired ablation, DEV seeds 0..29 (diff = ablated - intact, 25-test Holm FWER) ==");
    for ((name, out, g), ph) in rows.iter().zip(adj) {
        println!(
            "{name:<24} {out:<18} intact={:+8.2} ablated={:+8.2} diff={:+8.2} t={:+6.2} p={:.5} p_holm={:.5} {}",
            g.mean_b,
            g.mean_a,
            g.mean_diff,
            g.t,
            g.p_two_sided,
            ph,
            if ph < 0.05 {
                if g.mean_diff < 0.0 {
                    "NECESSARY"
                } else {
                    "HARMFUL"
                }
            } else {
                "not significant"
            }
        );
    }
    println!("\n== 12-Gate Suite & Invariant Verdicts Under Structural Ablation (DEV seeds 0..29) ==");
    for (name, pass_cnt, rv, iv) in suite_summaries {
        let failed_gates: Vec<String> = rv
            .iter()
            .filter(|r| !r.pass)
            .map(|r| format!("{} (t={:+.2}, p_holm={:.4})", r.name, r.stats.t, r.p_holm))
            .chain(
                iv.iter()
                    .filter(|i| !i.pass)
                    .map(|i| format!("{} ({})", i.name, i.detail)),
            )
            .collect();
        let fail_str = if failed_gates.is_empty() {
            "ALL 12/12 PASS".to_string()
        } else {
            format!("FAILED: {}", failed_gates.join(" ; "))
        };
        println!("{name:<24} {pass_cnt:>2}/12 | {fail_str}");
    }
}
