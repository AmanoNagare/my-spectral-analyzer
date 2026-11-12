use eframe::egui;
use rustfft::{num_complex::Complex, FftPlanner};
use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use windows::core::*;
use windows::Win32::Foundation::*;
use windows::Win32::Media::Audio::*;
use windows::Win32::Media::Multimedia::*;
use windows::Win32::System::Com::StructuredStorage::*;
use windows::Win32::System::Com::*;
use windows::Win32::System::Threading::*;
use windows::Win32::System::Variant::*;

const FFT_SIZE: usize = 1024;
const HOP_SIZE: usize = 512;
const NUM_POINTS: usize = 256;
const SCOPE_LEN: usize = 384;
const SCREEN_WIDTH: f32 = 2560.0;
const INITIAL_HEIGHT: f32 = 96.0;

#[derive(Clone)]
struct AnalyzerConfig {
    /// Decay rate per 0.1s (dB / 0.1s) - default 10.0 dB
    decay_db_per_100ms: f32,
    /// Peak hold decay rate per 0.1s (dB / 0.1s)
    hold_decay_db_per_100ms: f32,
    /// Input gain boost (dB)
    gain_db: f32,
    /// Display dynamic range (dB)
    dynamic_range_db: f32,
    /// Kaiser window beta (smaller = sharper/narrower peaks)
    kaiser_beta: f32,
    /// Spatial peak sharpening factor
    sharpen: f32,
    /// Background oscilloscope visibility and gain
    show_scope: bool,
    scope_gain: f32,
}

impl Default for AnalyzerConfig {
    fn default() -> Self {
        Self {
            decay_db_per_100ms: 10.0,
            hold_decay_db_per_100ms: 3.5,
            gain_db: 24.0,
            dynamic_range_db: 65.0,
            kaiser_beta: 1.8,
            sharpen: 0.25,
            show_scope: true,
            scope_gain: 1.5,
        }
    }
}

struct SharedState {
    spectrum: [f32; NUM_POINTS],
    peak_holds: [f32; NUM_POINTS],
    spectrum_db: [f32; NUM_POINTS],
    hold_db: [f32; NUM_POINTS],
    scope_samples: VecDeque<f32>,
    dominant_freq_hz: f32,
    dominant_ratio: f32,
    peak_level: f32,
    status_text: String,
    config: AnalyzerConfig,
}

fn main() -> eframe::Result<()> {
    let state = Arc::new(Mutex::new(SharedState {
        spectrum: [0.0f32; NUM_POINTS],
        peak_holds: [0.0f32; NUM_POINTS],
        spectrum_db: [-120.0f32; NUM_POINTS],
        hold_db: [-120.0f32; NUM_POINTS],
        scope_samples: VecDeque::from(vec![0.0f32; SCOPE_LEN]),
        dominant_freq_hz: 0.0,
        dominant_ratio: 0.0,
        peak_level: 0.0,
        status_text: "Initializing Process Loopback...".to_string(),
        config: AnalyzerConfig::default(),
    }));

    let state_clone = Arc::clone(&state);
    thread::spawn(move || {
        if let Err(e) = run_process_loopback(state_clone.clone()) {
            if let Ok(mut g) = state_clone.lock() {
                g.status_text = format!("Init Error: {}", e);
            }
        }
    });

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_position([0.0, 0.0])
            .with_inner_size([SCREEN_WIDTH, INITIAL_HEIGHT])
            .with_min_inner_size([640.0, 48.0])
            .with_always_on_top()
            .with_decorations(false)
            .with_transparent(true),
        ..Default::default()
    };

    eframe::run_native(
        "Optical Spectrum Analyzer",
        options,
        Box::new(|_cc| {
            Ok(Box::new(SpectrumApp {
                state,
                show_settings: false,
            }))
        }),
    )
}

#[inline]
fn optical_spectrum_color(ratio: f32, alpha: u8) -> egui::Color32 {
    let t = ratio.clamp(0.0, 1.0);
    let hue = t * 0.84; // Left: Red (0.0) -> Right: Violet/Magenta (0.84)
    let sat = 0.94;
    let h6 = hue * 6.0;
    let sector = h6 as i32;
    let f = h6 - sector as f32;
    let p = 1.0 - sat;
    let q = 1.0 - sat * f;
    let u = 1.0 - sat * (1.0 - f);

    let (r, g, b) = match sector {
        0 => (1.0, u, p),
        1 => (q, 1.0, p),
        2 => (p, 1.0, u),
        3 => (p, q, 1.0),
        4 => (u, p, 1.0),
        _ => (1.0, p, q),
    };

    egui::Color32::from_rgba_unmultiplied(
        (r * 255.0) as u8,
        (g * 255.0) as u8,
        (b * 255.0) as u8,
        alpha,
    )
}

fn bessel_i0(x: f32) -> f32 {
    let mut sum = 1.0f32;
    let mut term = 1.0f32;
    let half_x = x * 0.5;
    for k in 1..16 {
        term *= half_x / k as f32;
        sum += term * term;
    }
    sum
}

fn build_kaiser_window(size: usize, beta: f32) -> Vec<f32> {
    let inv_i0_beta = 1.0 / bessel_i0(beta);
    let half_len = (size as f32 - 1.0) * 0.5;
    (0..size)
        .map(|i| {
            let norm = (i as f32 - half_len) / half_len;
            let arg = beta * (1.0 - norm * norm).max(0.0).sqrt();
            bessel_i0(arg) * inv_i0_beta
        })
        .collect()
}

fn write_smooth_curve(
    values: &[f32; NUM_POINTS],
    rect: egui::Rect,
    top_padding: f32,
    bottom_padding: f32,
    subdivisions: usize,
    out: &mut Vec<(egui::Pos2, f32)>,
) {
    out.clear();
    let n = NUM_POINTS;
    let available_height = (rect.height() - top_padding - bottom_padding).max(10.0);
    let y_bottom = rect.bottom() - bottom_padding;
    let inv_n_minus_1 = 1.0 / (n - 1) as f32;
    let inv_sub = 1.0 / subdivisions as f32;

    for i in 0..(n - 1) {
        let p0 = if i == 0 { values[0] } else { values[i - 1] };
        let p1 = values[i];
        let p2 = values[i + 1];
        let p3 = if i + 2 < n { values[i + 2] } else { values[n - 1] };

        let c0 = 2.0 * p1;
        let c1 = -p0 + p2;
        let c2 = 2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3;
        let c3 = -p0 + 3.0 * p1 - 3.0 * p2 + p3;

        let steps = if i == n - 2 {
            subdivisions + 1
        } else {
            subdivisions
        };

        for s in 0..steps {
            let t = s as f32 * inv_sub;
            let t2 = t * t;
            let t3 = t2 * t;

            let interp_val = (0.5 * (c0 + c1 * t + c2 * t2 + c3 * t3)).clamp(0.0, 1.0);
            let global_ratio = (i as f32 + t) * inv_n_minus_1;
            let x = rect.left() + global_ratio * rect.width();
            let y = y_bottom - interp_val * available_height;

            out.push((egui::pos2(x, y), global_ratio));
        }
    }
}

struct SpectrumApp {
    state: Arc<Mutex<SharedState>>,
    show_settings: bool,
}

impl eframe::App for SpectrumApp {
    fn clear_color(&self, _visuals: &egui::Visuals) -> [f32; 4] {
        egui::Color32::from_rgba_unmultiplied(8, 10, 16, 220).to_normalized_gamma_f32()
    }

    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if ctx.input(|i| i.key_pressed(egui::Key::Escape)) {
            if self.show_settings {
                self.show_settings = false;
            } else {
                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            }
        }

        if !self.show_settings {
            let scroll_delta = ctx.input(|i| i.raw_scroll_delta.y);
            if scroll_delta != 0.0 {
                if let Some(inner_rect) = ctx.input(|i| i.viewport().inner_rect) {
                    let delta_h = if scroll_delta > 0.0 { 12.0 } else { -12.0 };
                    let new_h = (inner_rect.height() + delta_h).clamp(48.0, 480.0);
                    ctx.send_viewport_cmd(egui::ViewportCommand::InnerSize(egui::vec2(
                        SCREEN_WIDTH,
                        new_h,
                    )));
                }
            }
        }

        let (spectrum, peak_holds, scope, dom_hz, dom_ratio, peak, status, mut cfg) = {
            let g = self.state.lock().unwrap();
            (
                g.spectrum,
                g.peak_holds,
                g.scope_samples.clone(),
                g.dominant_freq_hz,
                g.dominant_ratio,
                g.peak_level,
                g.status_text.clone(),
                g.config.clone(),
            )
        };

        egui::CentralPanel::default()
            .frame(egui::Frame::none())
            .show(ctx, |ui| {
                let rect = ui.max_rect();

                // Top-right option button area (ASCII text "OPT")
                let btn_size = egui::vec2(38.0, 18.0);
                let btn_rect = egui::Rect::from_min_size(
                    egui::pos2(rect.right() - btn_size.x - 8.0, rect.top() + 4.0),
                    btn_size,
                );

                let drag_rect = egui::Rect::from_min_max(
                    rect.min,
                    egui::pos2(btn_rect.left() - 4.0, rect.bottom()),
                );
                let response = ui.interact(drag_rect, ui.id(), egui::Sense::drag());
                if response.dragged() {
                    ctx.send_viewport_cmd(egui::ViewportCommand::StartDrag);
                }

                let painter = ui.painter();

                // 1. Background Oscilloscope (Raw Waveform)
                if cfg.show_scope && scope.len() >= 2 {
                    let center_y = rect.center().y;
                    let amp_scale = rect.height() * 0.42 * cfg.scope_gain;
                    let step_x = rect.width() / (scope.len() - 1) as f32;
                    let max_amp = rect.height() * 0.46;

                    let scope_pts: Vec<egui::Pos2> = scope
                        .iter()
                        .enumerate()
                        .map(|(i, &s)| {
                            let x = rect.left() + i as f32 * step_x;
                            let clamped = (s * amp_scale).clamp(-max_amp, max_amp);
                            egui::pos2(x, center_y - clamped)
                        })
                        .collect();

                    painter.add(egui::Shape::line(
                        scope_pts,
                        egui::Stroke::new(
                            1.1_f32,
                            egui::Color32::from_rgba_unmultiplied(210, 230, 255, 75),
                        ),
                    ));
                }

                let top_padding = 16.0;
                let bottom_padding = 2.0;
                let y_bottom = rect.bottom() - bottom_padding;

                let mut curve_pts = Vec::with_capacity(NUM_POINTS * 3);
                write_smooth_curve(&spectrum, rect, top_padding, bottom_padding, 3, &mut curve_pts);

                if curve_pts.len() >= 2 {
                    let mut fill_mesh = egui::epaint::Mesh::default();
                    let mut line_mesh = egui::epaint::Mesh::default();
                    let half_thickness = 1.1_f32;

                    for (idx, &(top_pos, ratio)) in curve_pts.iter().enumerate() {
                        let top_col = optical_spectrum_color(ratio, 145);
                        let bot_col = optical_spectrum_color(ratio, 15);
                        let line_col = optical_spectrum_color(ratio, 250);

                        fill_mesh.vertices.push(egui::epaint::Vertex {
                            pos: top_pos,
                            uv: egui::epaint::WHITE_UV,
                            color: top_col,
                        });
                        fill_mesh.vertices.push(egui::epaint::Vertex {
                            pos: egui::pos2(top_pos.x, y_bottom),
                            uv: egui::epaint::WHITE_UV,
                            color: bot_col,
                        });

                        line_mesh.vertices.push(egui::epaint::Vertex {
                            pos: egui::pos2(top_pos.x, top_pos.y - half_thickness),
                            uv: egui::epaint::WHITE_UV,
                            color: line_col,
                        });
                        line_mesh.vertices.push(egui::epaint::Vertex {
                            pos: egui::pos2(top_pos.x, top_pos.y + half_thickness),
                            uv: egui::epaint::WHITE_UV,
                            color: line_col,
                        });

                        if idx > 0 {
                            let base = (idx as u32 - 1) * 2;
                            fill_mesh.indices.extend_from_slice(&[
                                base,
                                base + 1,
                                base + 2,
                                base + 1,
                                base + 3,
                                base + 2,
                            ]);
                            line_mesh.indices.extend_from_slice(&[
                                base,
                                base + 1,
                                base + 2,
                                base + 1,
                                base + 3,
                                base + 2,
                            ]);
                        }
                    }

                    painter.add(egui::Shape::mesh(fill_mesh));
                    painter.add(egui::Shape::mesh(line_mesh));
                }

                // 3. Peak Hold Trace
                let mut hold_pts = Vec::with_capacity(NUM_POINTS * 2);
                write_smooth_curve(&peak_holds, rect, top_padding, bottom_padding, 2, &mut hold_pts);
                if hold_pts.len() >= 2 {
                    let mut hold_mesh = egui::epaint::Mesh::default();
                    for (idx, &(pos, ratio)) in hold_pts.iter().enumerate() {
                        let col = optical_spectrum_color(ratio, 115);
                        hold_mesh.vertices.push(egui::epaint::Vertex {
                            pos: egui::pos2(pos.x, pos.y - 0.5),
                            uv: egui::epaint::WHITE_UV,
                            color: col,
                        });
                        hold_mesh.vertices.push(egui::epaint::Vertex {
                            pos: egui::pos2(pos.x, pos.y + 0.5),
                            uv: egui::epaint::WHITE_UV,
                            color: col,
                        });
                        if idx > 0 {
                            let base = (idx as u32 - 1) * 2;
                            hold_mesh.indices.extend_from_slice(&[
                                base,
                                base + 1,
                                base + 2,
                                base + 1,
                                base + 3,
                                base + 2,
                            ]);
                        }
                    }
                    painter.add(egui::Shape::mesh(hold_mesh));
                }

                // 4. Dominant Frequency (Hz) Tracking Badge
                if peak > 0.001 && dom_hz > 20.0 && !curve_pts.is_empty() {
                    let exact_idx = ((dom_ratio * (curve_pts.len() - 1) as f32) as usize)
                        .min(curve_pts.len() - 1);
                    let (peak_pos, peak_ratio) = curve_pts[exact_idx];
                    let peak_color = optical_spectrum_color(peak_ratio, 255);

                    painter.circle_filled(peak_pos, 4.5, optical_spectrum_color(peak_ratio, 90));
                    painter.circle_filled(peak_pos, 2.2, egui::Color32::WHITE);

                    let freq_label = if dom_hz >= 1000.0 {
                        format!("{:.2} kHz", dom_hz / 1000.0)
                    } else {
                        format!("{:.0} Hz", dom_hz)
                    };

                    let label_x = peak_pos.x.clamp(rect.left() + 36.0, rect.right() - 74.0);
                    let label_y = (peak_pos.y - 6.0).max(rect.top() + 14.0);

                    let badge_rect = egui::Rect::from_center_size(
                        egui::pos2(label_x, label_y - 6.0),
                        egui::vec2(56.0, 14.0),
                    );
                    painter.rect_filled(
                        badge_rect,
                        3.5,
                        egui::Color32::from_rgba_unmultiplied(10, 12, 20, 225),
                    );
                    painter.rect_stroke(
                        badge_rect,
                        3.5,
                        egui::Stroke::new(1.0_f32, peak_color),
                    );

                    painter.text(
                        egui::pos2(label_x, label_y - 0.5),
                        egui::Align2::CENTER_BOTTOM,
                        freq_label,
                        egui::FontId::monospace(10.0),
                        egui::Color32::WHITE,
                    );
                }

                // 5. Top-left HUD Status (ASCII only)
                let hud = format!(
                    "{} | Decay: -{:.1}dB/0.1s | Gain: +{:.0}dB | Beta: {:.1}",
                    status, cfg.decay_db_per_100ms, cfg.gain_db, cfg.kaiser_beta
                );
                painter.text(
                    egui::pos2(rect.left() + 10.0, rect.top() + 3.0),
                    egui::Align2::LEFT_TOP,
                    hud,
                    egui::FontId::monospace(9.5),
                    egui::Color32::from_rgba_unmultiplied(170, 190, 220, 140),
                );

                // 6. Top-right Option Button ("OPT")
                let btn_resp = ui.interact(btn_rect, ui.id().with("opt_btn"), egui::Sense::click());
                let bg_alpha = if self.show_settings || btn_resp.hovered() {
                    235
                } else {
                    160
                };
                let border_col = if self.show_settings {
                    egui::Color32::from_rgb(0, 230, 255)
                } else {
                    egui::Color32::from_rgba_unmultiplied(140, 170, 210, 140)
                };

                painter.rect_filled(
                    btn_rect,
                    3.5,
                    egui::Color32::from_rgba_unmultiplied(18, 24, 38, bg_alpha),
                );
                painter.rect_stroke(btn_rect, 3.5, egui::Stroke::new(1.0_f32, border_col));
                painter.text(
                    btn_rect.center(),
                    egui::Align2::CENTER_CENTER,
                    "OPT",
                    egui::FontId::monospace(10.5),
                    egui::Color32::WHITE,
                );

                if btn_resp.clicked() {
                    self.show_settings = !self.show_settings;
                    if self.show_settings && rect.height() < 260.0 {
                        ctx.send_viewport_cmd(egui::ViewportCommand::InnerSize(egui::vec2(
                            SCREEN_WIDTH,
                            270.0,
                        )));
                    } else if !self.show_settings && rect.height() == 270.0 {
                        ctx.send_viewport_cmd(egui::ViewportCommand::InnerSize(egui::vec2(
                            SCREEN_WIDTH,
                            INITIAL_HEIGHT,
                        )));
                    }
                }
            });

        // 7. Settings Window (All English)
        if self.show_settings {
            let mut changed = false;
            egui::Window::new("Spectrum Parameters")
                .anchor(egui::Align2::RIGHT_TOP, egui::vec2(-10.0, 26.0))
                .collapsible(false)
                .resizable(false)
                .fixed_size(egui::vec2(350.0, 220.0))
                .frame(
                    egui::Frame::window(&ctx.style())
                        .fill(egui::Color32::from_rgba_unmultiplied(14, 18, 28, 245))
                        .stroke(egui::Stroke::new(
                            1.0_f32,
                            egui::Color32::from_rgba_unmultiplied(80, 180, 255, 160),
                        )),
                )
                .show(ctx, |ui| {
                    egui::Grid::new("config_grid")
                        .num_columns(2)
                        .spacing([12.0, 6.0])
                        .show(ui, |ui| {
                            ui.label("Decay Rate (dB / 0.1s):");
                            changed |= ui
                                .add(
                                    egui::Slider::new(&mut cfg.decay_db_per_100ms, 1.0..=40.0)
                                        .suffix(" dB"),
                                )
                                .changed();
                            ui.end_row();

                            ui.label("Hold Decay (dB / 0.1s):");
                            changed |= ui
                                .add(
                                    egui::Slider::new(
                                        &mut cfg.hold_decay_db_per_100ms,
                                        0.5..=25.0,
                                    )
                                    .suffix(" dB"),
                                )
                                .changed();
                            ui.end_row();

                            ui.label("Input Gain Boost:");
                            changed |= ui
                                .add(
                                    egui::Slider::new(&mut cfg.gain_db, 0.0..=50.0)
                                        .prefix("+")
                                        .suffix(" dB"),
                                )
                                .changed();
                            ui.end_row();

                            ui.label("Dynamic Range:");
                            changed |= ui
                                .add(
                                    egui::Slider::new(&mut cfg.dynamic_range_db, 30.0..=100.0)
                                        .suffix(" dB"),
                                )
                                .changed();
                            ui.end_row();

                            ui.label("Window Kaiser Beta:");
                            changed |= ui
                                .add(egui::Slider::new(&mut cfg.kaiser_beta, 0.0..=8.0))
                                .changed();
                            ui.end_row();

                            ui.label("Peak Sharpening:");
                            changed |= ui
                                .add(egui::Slider::new(&mut cfg.sharpen, 0.0..=0.6))
                                .changed();
                            ui.end_row();

                            ui.label("Oscilloscope:");
                            ui.horizontal(|ui| {
                                changed |= ui.checkbox(&mut cfg.show_scope, "Show").changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut cfg.scope_gain, 0.2..=5.0)
                                            .prefix("x"),
                                    )
                                    .changed();
                            });
                            ui.end_row();
                        });

                    ui.separator();
                    ui.horizontal(|ui| {
                        if ui.button("Reset Default (10dB/0.1s)").clicked() {
                            cfg = AnalyzerConfig::default();
                            changed = true;
                        }
                        if ui.button("Close").clicked() {
                            self.show_settings = false;
                            ctx.send_viewport_cmd(egui::ViewportCommand::InnerSize(egui::vec2(
                                SCREEN_WIDTH,
                                INITIAL_HEIGHT,
                            )));
                        }
                    });
                });

            if changed {
                if let Ok(mut g) = self.state.lock() {
                    g.config = cfg;
                }
            }
        }

        ctx.request_repaint();
    }
}

#[implement(IActivateAudioInterfaceCompletionHandler)]
struct ActivationHandler {
    done: Arc<(Mutex<bool>, Condvar)>,
    result: Mutex<Option<Result<IAudioClient>>>,
}

impl IActivateAudioInterfaceCompletionHandler_Impl for ActivationHandler_Impl {
    fn ActivateCompleted(
        &self,
        activateoperation: Ref<'_, IActivateAudioInterfaceAsyncOperation>,
    ) -> Result<()> {
        let res = (|| -> Result<IAudioClient> {
            let op = activateoperation.ok()?;
            let mut hr = HRESULT(0);
            let mut unk: Option<IUnknown> = None;
            unsafe {
                op.GetActivateResult(&mut hr, &mut unk)?;
            }
            hr.ok()?;
            let client: IAudioClient = unk.ok_or(Error::from(E_NOINTERFACE))?.cast()?;
            Ok(client)
        })();

        if let Ok(mut guard) = self.result.lock() {
            *guard = Some(res);
        }

        let (lock, cvar) = &*self.done;
        if let Ok(mut finished) = lock.lock() {
            *finished = true;
            cvar.notify_all();
        }
        Ok(())
    }
}

fn run_process_loopback(state: Arc<Mutex<SharedState>>) -> Result<()> {
    unsafe {
        CoInitializeEx(None, COINIT_MULTITHREADED).ok()?;

        let current_pid = GetCurrentProcessId();
        let mut params = AUDIOCLIENT_ACTIVATION_PARAMS {
            ActivationType: AUDIOCLIENT_ACTIVATION_TYPE_PROCESS_LOOPBACK,
            Anonymous: AUDIOCLIENT_ACTIVATION_PARAMS_0 {
                ProcessLoopbackParams: AUDIOCLIENT_PROCESS_LOOPBACK_PARAMS {
                    TargetProcessId: current_pid,
                    ProcessLoopbackMode: PROCESS_LOOPBACK_MODE_EXCLUDE_TARGET_PROCESS_TREE,
                },
            },
        };

        let mut prop_variant = PROPVARIANT::default();
        (*prop_variant.Anonymous.Anonymous).vt = VT_BLOB;
        (*prop_variant.Anonymous.Anonymous).Anonymous.blob = BLOB {
            cbSize: std::mem::size_of::<AUDIOCLIENT_ACTIVATION_PARAMS>() as u32,
            pBlobData: &mut params as *mut _ as *mut u8,
        };

        let done = Arc::new((Mutex::new(false), Condvar::new()));
        let handler = ActivationHandler {
            done: Arc::clone(&done),
            result: Mutex::new(None),
        };
        let handler_interface: IActivateAudioInterfaceCompletionHandler = handler.into();

        let _op = ActivateAudioInterfaceAsync(
            VIRTUAL_AUDIO_DEVICE_PROCESS_LOOPBACK,
            &IAudioClient::IID,
            Some(&prop_variant),
            &handler_interface,
        )?;

        let (lock, cvar) = &*done;
        let mut finished = lock.lock().unwrap();
        while !*finished {
            let res = cvar.wait_timeout(finished, Duration::from_secs(5)).unwrap();
            finished = res.0;
            if res.1.timed_out() {
                break;
            }
        }

        let handler_impl: &ActivationHandler = handler_interface.as_impl();
        let audio_client = handler_impl
            .result
            .lock()
            .unwrap()
            .take()
            .ok_or(Error::from(E_FAIL))??;

        let sample_rate = 48000u32;
        let channels = 2u16;
        let bits_per_sample = 32u16;
        let block_align = channels * (bits_per_sample / 8);

        let wave_format = WAVEFORMATEX {
            wFormatTag: WAVE_FORMAT_IEEE_FLOAT as u16,
            nChannels: channels,
            nSamplesPerSec: sample_rate,
            nAvgBytesPerSec: sample_rate * block_align as u32,
            nBlockAlign: block_align,
            wBitsPerSample: bits_per_sample,
            cbSize: 0,
        };

        audio_client.Initialize(
            AUDCLNT_SHAREMODE_SHARED,
            AUDCLNT_STREAMFLAGS_LOOPBACK,
            100_000,
            0,
            &wave_format,
            None,
        )?;

        let capture_client: IAudioCaptureClient = audio_client.GetService()?;

        if let Ok(mut g) = state.lock() {
            g.status_text = "OPTICAL SPECTRUM (48kHz)".to_string();
        }

        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(FFT_SIZE);

        let mut ring_buf = [0.0f32; FFT_SIZE];
        let mut ring_write_idx = 0usize;
        let mut samples_since_last_fft = 0usize;
        let mut fft_buf = vec![Complex::new(0.0f32, 0.0f32); FFT_SIZE];
        let mut raw_db = [-120.0f32; NUM_POINTS];
        let mut sharp_db = [-120.0f32; NUM_POINTS];

        let mut current_beta = AnalyzerConfig::default().kaiser_beta;
        let mut window = build_kaiser_window(FFT_SIZE, current_beta);

        let half_fft = FFT_SIZE / 2;
        let bin_lookup: Vec<(usize, usize, f32, f32)> = (0..NUM_POINTS)
            .map(|p| {
                let ratio = p as f32 / (NUM_POINTS - 1) as f32;
                let mapped = ratio.powf(1.85);
                let exact_idx = 1.0 + mapped * ((half_fft - 2) as f32);
                let idx0 = exact_idx.floor() as usize;
                let idx1 = (idx0 + 1).min(half_fft - 1);
                let frac = exact_idx - idx0 as f32;
                (idx0, idx1, frac, ratio)
            })
            .collect();

        let min_bin = (30.0 * FFT_SIZE as f32 / sample_rate as f32) as usize;
        let max_bin = ((20000.0 * FFT_SIZE as f32 / sample_rate as f32) as usize).min(half_fft - 1);
        let inv_fft_norm = 2.0 / FFT_SIZE as f32;
        let bin_to_hz = sample_rate as f32 / FFT_SIZE as f32;

        audio_client.Start()?;

        let mut scope_downsample_counter = 0usize;
        let mut last_tick = Instant::now();

        loop {
            thread::sleep(Duration::from_millis(5));

            let now = Instant::now();
            let dt_sec = now.duration_since(last_tick).as_secs_f32().min(0.1);
            last_tick = now;

            let cfg = state.lock().map(|g| g.config.clone()).unwrap_or_default();

            if (cfg.kaiser_beta - current_beta).abs() > 0.01 {
                current_beta = cfg.kaiser_beta;
                window = build_kaiser_window(FFT_SIZE, current_beta);
            }

            let decay_db = (cfg.decay_db_per_100ms * 10.0) * dt_sec;
            let hold_decay_db = (cfg.hold_decay_db_per_100ms * 10.0) * dt_sec;
            let floor_db = -cfg.dynamic_range_db;

            let mut packet_len = capture_client.GetNextPacketSize()?;
            if packet_len == 0 {
                if let Ok(mut g) = state.lock() {
                    for i in 0..NUM_POINTS {
                        g.spectrum_db[i] = (g.spectrum_db[i] - decay_db).max(floor_db - 20.0);
                        g.hold_db[i] = (g.hold_db[i] - hold_decay_db).max(floor_db - 20.0);

                        g.spectrum[i] =
                            ((g.spectrum_db[i] - floor_db) / cfg.dynamic_range_db).clamp(0.0, 1.0);
                        g.peak_holds[i] =
                            ((g.hold_db[i] - floor_db) / cfg.dynamic_range_db).clamp(0.0, 1.0);
                    }
                    g.peak_level *= 0.55;
                    for _ in 0..12 {
                        g.scope_samples.pop_front();
                        g.scope_samples.push_back(0.0);
                    }
                }
                continue;
            }

            let mut local_peak = 0.0f32;
            let mut fft_updated = false;
            let mut latest_dom_hz = 0.0f32;
            let mut latest_dom_ratio = 0.0f32;

            while packet_len > 0 {
                let mut data_ptr: *mut u8 = std::ptr::null_mut();
                let mut num_frames = 0u32;
                let mut flags = 0u32;

                capture_client.GetBuffer(
                    &mut data_ptr,
                    &mut num_frames,
                    &mut flags,
                    None,
                    None,
                )?;

                if num_frames > 0 && !data_ptr.is_null() {
                    let is_silent = (flags & (AUDCLNT_BUFFERFLAGS_SILENT.0 as u32)) != 0;
                    if !is_silent {
                        let float_slice = std::slice::from_raw_parts(
                            data_ptr as *const f32,
                            (num_frames as usize) * (channels as usize),
                        );

                        for frame in float_slice.chunks(channels as usize) {
                            let mono = (frame[0] + frame[1]) * 0.5;
                            if mono.is_finite() {
                                let abs_m = mono.abs();
                                if abs_m > local_peak {
                                    local_peak = abs_m;
                                }

                                ring_buf[ring_write_idx] = mono;
                                ring_write_idx = (ring_write_idx + 1) % FFT_SIZE;
                                samples_since_last_fft += 1;

                                scope_downsample_counter += 1;
                                if scope_downsample_counter % 2 == 0 {
                                    if let Ok(mut g) = state.lock() {
                                        if g.scope_samples.len() >= SCOPE_LEN {
                                            g.scope_samples.pop_front();
                                        }
                                        g.scope_samples.push_back(mono);
                                    }
                                }
                            }

                            if samples_since_last_fft >= HOP_SIZE {
                                samples_since_last_fft = 0;
                                fft_updated = true;

                                for i in 0..FFT_SIZE {
                                    let s = ring_buf[(ring_write_idx + i) % FFT_SIZE];
                                    fft_buf[i] = Complex::new(s * window[i], 0.0);
                                }

                                fft.process(&mut fft_buf);

                                let mut global_max_bin = min_bin;
                                let mut global_max_mag_sq = 0.0f32;
                                for idx in min_bin..=max_bin {
                                    let m_sq = fft_buf[idx].norm_sqr();
                                    if m_sq > global_max_mag_sq {
                                        global_max_mag_sq = m_sq;
                                        global_max_bin = idx;
                                    }
                                }

                                let exact_bin =
                                    if global_max_bin > 1 && global_max_bin + 1 < half_fft {
                                        let y0 = fft_buf[global_max_bin - 1].norm();
                                        let y1 = fft_buf[global_max_bin].norm();
                                        let y2 = fft_buf[global_max_bin + 1].norm();
                                        let denom = y0 - 2.0 * y1 + y2;
                                        if denom.abs() > 1e-6 {
                                            global_max_bin as f32 + 0.5 * (y0 - y2) / denom
                                        } else {
                                            global_max_bin as f32
                                        }
                                    } else {
                                        global_max_bin as f32
                                    };
                                latest_dom_hz = exact_bin * bin_to_hz;

                                let mut max_db_val = -200.0f32;
                                for (p, &(idx0, idx1, frac, ratio)) in
                                    bin_lookup.iter().enumerate()
                                {
                                    let m0 = fft_buf[idx0].norm();
                                    let m1 = fft_buf[idx1].norm();
                                    let mag = ((m0 * (1.0 - frac) + m1 * frac) * inv_fft_norm)
                                        .max(1e-7);
                                    let db = 20.0 * mag.log10() + cfg.gain_db;
                                    raw_db[p] = db;

                                    if db > max_db_val {
                                        max_db_val = db;
                                        latest_dom_ratio = ratio;
                                    }
                                }

                                sharp_db[0] = raw_db[0];
                                sharp_db[NUM_POINTS - 1] = raw_db[NUM_POINTS - 1];
                                let side_w = cfg.sharpen * 0.5;
                                let center_w = 1.0 + cfg.sharpen;
                                for p in 1..(NUM_POINTS - 1) {
                                    sharp_db[p] = raw_db[p] * center_w
                                        - (raw_db[p - 1] + raw_db[p + 1]) * side_w;
                                }
                            }
                        }
                    }
                }

                capture_client.ReleaseBuffer(num_frames)?;
                packet_len = capture_client.GetNextPacketSize()?;
            }

            if let Ok(mut g) = state.lock() {
                for i in 0..NUM_POINTS {
                    let decayed = g.spectrum_db[i] - decay_db;
                    let target = if fft_updated {
                        sharp_db[i].max(decayed)
                    } else {
                        decayed
                    };
                    g.spectrum_db[i] = target.max(floor_db - 20.0);

                    let hold_decayed = g.hold_db[i] - hold_decay_db;
                    g.hold_db[i] = g.spectrum_db[i].max(hold_decayed).max(floor_db - 20.0);

                    g.spectrum[i] =
                        ((g.spectrum_db[i] - floor_db) / cfg.dynamic_range_db).clamp(0.0, 1.0);
                    g.peak_holds[i] =
                        ((g.hold_db[i] - floor_db) / cfg.dynamic_range_db).clamp(0.0, 1.0);
                }

                if fft_updated {
                    g.dominant_freq_hz = g.dominant_freq_hz * 0.25 + latest_dom_hz * 0.75;
                    g.dominant_ratio = g.dominant_ratio * 0.25 + latest_dom_ratio * 0.75;
                }
                g.peak_level = g.peak_level * 0.4 + local_peak * 0.6;
            }
        }
    }
}