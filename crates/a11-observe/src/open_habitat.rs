//! Shared whole-body open habitat for the `alpha11-life` judges (`energy` v2 and `teach`).
//!
//! Mirrors the live `OpenSocialHabitat` (Earth ecology, Elder in slot 0, infant body in slot 1)
//! with two deliberate differences, both pre-registered in ledger entry 57:
//! - The Elder runs `ControlMode::FrozenPlasticity` from birth, so its policy cannot change with
//!   the organism code under test (fixes the v1 confound found in ledger entry 56). A reference
//!   fingerprint ([`elder_reference_hash`]) detects any code change that alters frozen dynamics.
//! - An optional environment-side caregiver reply frame can be mixed into the infant's ears, and
//!   food can be provisioned through `MultiGenTerrarium::provision_open_pod_near`. Neither touches
//!   organism internals; the organism only experiences ordinary sound and `metabolic_influx`.

use a11_core::{Alpha11Organism, OrganismConfig};
use a11_medium::{ControlMode, FieldPulse, FieldView, OrganismEngine, WorldMedium};
use a11_portals::{EMBODIED_ORGAN_ACTUATORS, EMBODIED_ORGAN_CHANNELS, EmbodiedBodyOrgans};
use a11_worlds::{
    EARTH_AUDIO_FRAME_SAMPLES, MultiGenTerrarium, TERRARIUM_ACTUATORS, TERRARIUM_CHANNELS,
};

pub const OPEN_CH: usize = TERRARIUM_CHANNELS + EMBODIED_ORGAN_CHANNELS;
pub const OPEN_ACT: usize = TERRARIUM_ACTUATORS + EMBODIED_ORGAN_ACTUATORS;
pub const FRAME: usize = EARTH_AUDIO_FRAME_SAMPLES;
const ELDER_WARMUP_TICKS: usize = 640;

#[inline]
fn clamp_unit(v: f32) -> f32 {
    if v.is_finite() { v.clamp(-1.0, 1.0) } else { 0.0 }
}

#[inline]
fn clamp_nonneg(v: f32) -> f32 {
    if v.is_finite() && v > 0.0 { v } else { 0.0 }
}

pub struct OpenHabitat {
    pub terrarium: MultiGenTerrarium,
    pub elder_brain: Alpha11Organism,
    pub elder_body: EmbodiedBodyOrgans,
    elder_ambient: [f32; OPEN_CH],
    elder_influx: f32,
    elder_stress: f32,
    pub focal_body: EmbodiedBodyOrgans,
    focal_ambient: [f32; OPEN_CH],
    focal_influx: f32,
    focal_stress: f32,
    reply_frame: Option<[f32; FRAME]>,
    last_elder_pulse: Vec<f32>,
}

impl OpenHabitat {
    #[must_use]
    pub fn new(seed: u64) -> Self {
        let mut terrarium = MultiGenTerrarium::new_earth_ecology(seed, 1);
        let mut elder_brain =
            Alpha11Organism::new(seed ^ 0xE1DE_A110_2026_0001, OrganismConfig::default());
        elder_brain.set_control_mode(ControlMode::FrozenPlasticity);
        let mut elder_body = EmbodiedBodyOrgans::new_adult(seed ^ 0xE1DE_B0D1_2026_0002);
        let mut elder_ambient = [0.0_f32; OPEN_CH];
        for _ in 0..ELDER_WARMUP_TICKS {
            let tv = terrarium.agent_view(0);
            for (i, slot) in elder_ambient[..TERRARIUM_CHANNELS].iter_mut().enumerate() {
                *slot = clamp_unit(tv.ambient_in.get(i).copied().unwrap_or(0.0));
            }
            elder_ambient[TERRARIUM_CHANNELS..].copy_from_slice(elder_body.sensory_channels());
            let view = FieldView {
                ambient_in: &elder_ambient,
                metabolic_influx: clamp_nonneg(tv.metabolic_influx),
                tissue_stress: clamp_nonneg(tv.tissue_stress + elder_body.last_organ_stress()),
                num_actuators: OPEN_ACT,
            };
            let ep = elder_brain
                .step(view)
                .unwrap_or_else(|_| FieldPulse::silent(OPEN_ACT));
            let mut eb = [0.0_f32; EMBODIED_ORGAN_ACTUATORS];
            for (j, slot) in eb.iter_mut().enumerate() {
                *slot = clamp_unit(
                    ep.actuator_out
                        .get(TERRARIUM_ACTUATORS + j)
                        .copied()
                        .unwrap_or(0.0),
                );
            }
            let _ = elder_body.step_limbs_and_radiate_voice(&eb);
            let (es, eh) = elder_body.embodied_kinematics();
            terrarium.set_agent_embodied_kinematics(0, es, eh);
            terrarium.set_agent_embodied_voice_rms(0, elder_body.vocal().last_radiated_rms());
            let mut ew = FieldPulse {
                actuator_out: Vec::with_capacity(TERRARIUM_ACTUATORS),
            };
            for i in 0..TERRARIUM_ACTUATORS {
                ew.actuator_out
                    .push(clamp_unit(ep.actuator_out.get(i).copied().unwrap_or(0.0)));
            }
            terrarium.apply_population_pulses(&[ew]);
            elder_body.apply_terrain_sway(terrarium.agent_lateral_sway(0));
            let mut env_l = [0.0_f32; FRAME];
            let mut env_r = [0.0_f32; FRAME];
            let enc = terrarium.binaural_earth_waves_for_agent(0, &mut env_l, &mut env_r);
            elder_body.finish_binaural_ecology_transduction(None, Some((&env_l, &env_r)), enc);
        }
        elder_body.vocal_mut().reset_window_observations();
        terrarium.respawn_newborn_in_slot(1);
        let focal_body = EmbodiedBodyOrgans::new_infant(seed ^ 0xF0CA_1B0D_2026_0003);
        let mut hab = Self {
            terrarium,
            elder_brain,
            elder_body,
            elder_ambient,
            elder_influx: 0.0,
            elder_stress: 0.0,
            focal_body,
            focal_ambient: [0.0; OPEN_CH],
            focal_influx: 0.0,
            focal_stress: 0.0,
            reply_frame: None,
            last_elder_pulse: vec![0.0; OPEN_ACT],
        };
        hab.refresh_views();
        hab
    }

    fn refresh_views(&mut self) {
        let ev = self.terrarium.agent_view(0);
        for (i, slot) in self.elder_ambient[..TERRARIUM_CHANNELS]
            .iter_mut()
            .enumerate()
        {
            *slot = clamp_unit(ev.ambient_in.get(i).copied().unwrap_or(0.0));
        }
        self.elder_ambient[TERRARIUM_CHANNELS..].copy_from_slice(self.elder_body.sensory_channels());
        self.elder_influx = clamp_nonneg(ev.metabolic_influx);
        self.elder_stress = clamp_nonneg(ev.tissue_stress + self.elder_body.last_organ_stress());

        let fv = self.terrarium.agent_view(1);
        for (i, slot) in self.focal_ambient[..TERRARIUM_CHANNELS]
            .iter_mut()
            .enumerate()
        {
            *slot = clamp_unit(fv.ambient_in.get(i).copied().unwrap_or(0.0));
        }
        self.focal_ambient[TERRARIUM_CHANNELS..].copy_from_slice(self.focal_body.sensory_channels());
        self.focal_influx = clamp_nonneg(fv.metabolic_influx);
        self.focal_stress = clamp_nonneg(fv.tissue_stress + self.focal_body.last_organ_stress());
    }

    /// Advances infant vocal-tract ontogeny from the organism's own maturation (as in `a11-live`).
    pub fn sync_vocal_ontogeny(&mut self, org: &Alpha11Organism) {
        let s = org.snapshot();
        let items = org.motor_maturation_by_channel();
        let mut sum = 0.0_f64;
        let mut n = 0_u32;
        for &(ch, mastery, _) in &items {
            if (12..17).contains(&ch) {
                sum += f64::from(mastery);
                n += 1;
            }
        }
        let voc_m = if n > 0 { (sum / f64::from(n)) as f32 } else { 0.0 };
        self.focal_body
            .advance_vocal_ontogeny(s.tick, s.mitosis_births, voc_m);
    }

    /// Queues one caregiver acoustic frame to be heard by the infant on the next `apply_pulse`,
    /// arriving from the Elder's direction (mixed with the Elder's own voice).
    pub fn set_reply_frame(&mut self, frame: Option<[f32; FRAME]>) {
        self.reply_frame = frame;
    }

    /// Infant position (slot 1).
    #[must_use]
    pub fn focal_pos(&self) -> (f32, f32) {
        self.terrarium
            .metrics()
            .agent_positions
            .get(1)
            .copied()
            .unwrap_or((0.0, 0.0))
    }

    /// Latest Elder motor pulse (used for the frozen-dynamics fingerprint).
    #[must_use]
    pub fn last_elder_pulse(&self) -> &[f32] {
        &self.last_elder_pulse
    }
}

impl WorldMedium for OpenHabitat {
    fn num_ambient_channels(&self) -> usize {
        OPEN_CH
    }

    fn num_actuators(&self) -> usize {
        OPEN_ACT
    }

    fn current_view(&self) -> FieldView<'_> {
        FieldView {
            ambient_in: &self.focal_ambient,
            metabolic_influx: self.focal_influx,
            tissue_stress: self.focal_stress,
            num_actuators: OPEN_ACT,
        }
    }

    fn apply_pulse(&mut self, focal_pulse: &FieldPulse) {
        let elder_view = FieldView {
            ambient_in: &self.elder_ambient,
            metabolic_influx: self.elder_influx,
            tissue_stress: self.elder_stress,
            num_actuators: OPEN_ACT,
        };
        let elder_pulse = self
            .elder_brain
            .step(elder_view)
            .unwrap_or_else(|_| FieldPulse::silent(OPEN_ACT));
        for (dst, &v) in self
            .last_elder_pulse
            .iter_mut()
            .zip(elder_pulse.actuator_out.iter())
        {
            *dst = v;
        }

        let mut eb = [0.0_f32; EMBODIED_ORGAN_ACTUATORS];
        let mut fb = [0.0_f32; EMBODIED_ORGAN_ACTUATORS];
        for j in 0..EMBODIED_ORGAN_ACTUATORS {
            eb[j] = clamp_unit(
                elder_pulse
                    .actuator_out
                    .get(TERRARIUM_ACTUATORS + j)
                    .copied()
                    .unwrap_or(0.0),
            );
            fb[j] = clamp_unit(
                focal_pulse
                    .actuator_out
                    .get(TERRARIUM_ACTUATORS + j)
                    .copied()
                    .unwrap_or(0.0),
            );
        }
        let elder_wave = self.elder_body.step_limbs_and_radiate_voice(&eb);
        let focal_wave = self.focal_body.step_limbs_and_radiate_voice(&fb);

        let (e_stab, e_hw) = self.elder_body.embodied_kinematics();
        let (f_stab, f_hw) = self.focal_body.embodied_kinematics();
        self.terrarium.set_agent_embodied_kinematics(0, e_stab, e_hw);
        self.terrarium.set_agent_embodied_kinematics(1, f_stab, f_hw);
        self.terrarium
            .set_agent_embodied_voice_rms(0, self.elder_body.vocal().last_radiated_rms());
        self.terrarium
            .set_agent_embodied_voice_rms(1, self.focal_body.vocal().last_radiated_rms());

        let mut ew = FieldPulse {
            actuator_out: Vec::with_capacity(TERRARIUM_ACTUATORS),
        };
        let mut fw = FieldPulse {
            actuator_out: Vec::with_capacity(TERRARIUM_ACTUATORS),
        };
        for i in 0..TERRARIUM_ACTUATORS {
            ew.actuator_out
                .push(clamp_unit(elder_pulse.actuator_out.get(i).copied().unwrap_or(0.0)));
            fw.actuator_out
                .push(clamp_unit(focal_pulse.actuator_out.get(i).copied().unwrap_or(0.0)));
        }
        self.terrarium.apply_population_pulses(&[ew, fw]);

        self.elder_body
            .apply_terrain_sway(self.terrarium.agent_lateral_sway(0));
        self.focal_body
            .apply_terrain_sway(self.terrarium.agent_lateral_sway(1));

        let positions = &self.terrarium.metrics().agent_positions;
        let e_pos = positions.first().copied().unwrap_or((0.0, 0.0));
        let f_pos = positions.get(1).copied().unwrap_or((0.0, 0.0));
        let rel_ef = (e_pos.0 - f_pos.0, e_pos.1 - f_pos.1);
        let rel_fe = (f_pos.0 - e_pos.0, f_pos.1 - e_pos.1);
        let pair_dist = rel_ef.0.hypot(rel_ef.1);
        if pair_dist <= 3.6 {
            let coupling = 0.26 * (1.0 - pair_dist / 3.6);
            EmbodiedBodyOrgans::couple_shared_hands(
                &mut self.elder_body,
                &mut self.focal_body,
                coupling,
            );
        }

        let peer_diff = self.terrarium.acoustic_transmittance(f_pos, e_pos);
        let mut elder_wave_diff = elder_wave;
        let mut focal_wave_diff = focal_wave;
        if peer_diff < 0.999 {
            for s in &mut elder_wave_diff {
                *s *= peer_diff;
            }
            for s in &mut focal_wave_diff {
                *s *= peer_diff;
            }
        }
        if let Some(reply) = self.reply_frame.take() {
            for (dst, &r) in elder_wave_diff.iter_mut().zip(reply.iter()) {
                *dst = clamp_unit(*dst + r);
            }
        }
        let mut f_l = [0.0_f32; FRAME];
        let mut f_r = [0.0_f32; FRAME];
        let f_enc = self
            .terrarium
            .binaural_earth_waves_for_agent(1, &mut f_l, &mut f_r);
        self.focal_body.finish_binaural_ecology_transduction(
            Some((&elder_wave_diff, rel_ef)),
            Some((&f_l, &f_r)),
            f_enc,
        );
        let mut e_l = [0.0_f32; FRAME];
        let mut e_r = [0.0_f32; FRAME];
        let e_enc = self
            .terrarium
            .binaural_earth_waves_for_agent(0, &mut e_l, &mut e_r);
        self.elder_body.finish_binaural_ecology_transduction(
            Some((&focal_wave_diff, rel_fe)),
            Some((&e_l, &e_r)),
            e_enc,
        );
        self.refresh_views();
    }
}

/// FNV-1a fingerprint of the frozen Elder's motor output over `ticks` steps in a habitat whose
/// infant emits silence. Changes only if code alters frozen-mode organism or world dynamics.
#[must_use]
pub fn elder_reference_hash(seed: u64, ticks: usize) -> u64 {
    let mut hab = OpenHabitat::new(seed);
    let silent = FieldPulse::silent(OPEN_ACT);
    let mut h: u64 = 0xCBF2_9CE4_8422_2325;
    for _ in 0..ticks {
        hab.apply_pulse(&silent);
        for &v in hab.last_elder_pulse() {
            for b in v.to_bits().to_le_bytes() {
                h ^= u64::from(b);
                h = h.wrapping_mul(0x0100_0000_01B3);
            }
        }
    }
    h
}
