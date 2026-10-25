// Absolute luminance (nits) measurement and HDR scopes for 10-bit 4:2:0 frames.
//
// Always, per frame, a fine log-nits histogram of max(R,G,B) goes to `cll`: it
// gives a MaxCLL that ignores a few stray pixels, and counts pixels at the ceiling.
//
// The frame is uploaded as packed u16 planes (Y, then U, then V), two samples per
// u32 word. Each thread handles one 2x2 pixel block (one chroma sample), so a
// 16x16 workgroup covers 32x32 pixels. Per workgroup we write peak/sum of
// max(R,G,B) nits (what MaxCLL/MaxFALL use) and of luminance. Optionally, per
// frame, we accumulate scopes into `scopes` at offsets given in Params:
//   waveform:    [R,G,B][row][col] counts, rows = log nits
//   histogram:   [L,R,G,B][bin] counts, bins = log nits
//   CIE xy:      [y bin][x bin] counts of chromaticity (x 0..0.8, y 0..0.9)
//   vectorscope: [Cr bin][Cb bin] counts of chroma (-0.5..0.5 each)

const SCOPE_WAVE: u32 = 1u;
const SCOPE_HIST: u32 = 2u;
const SCOPE_CIE: u32 = 4u;
const SCOPE_VEC: u32 = 8u;
const HIST_BINS: u32 = 256u;
const XY_BINS: u32 = 256u;
const CLL_BINS: u32 = 1024u;

struct Params {
    width: u32,
    height: u32,
    u_off: u32,          // index (in u16 samples) of the U plane
    v_off: u32,          // index of the V plane
    wave_cols: u32,
    wave_rows: u32,
    scope_flags: u32,
    wg_base: u32,        // offset into wg_stats for this frame (in f32s)
    scope_base: u32,     // offset into scopes for this frame (in u32s)
    hist_off: u32,       // offsets of each scope relative to scope_base
    cie_off: u32,
    vec_off: u32,
    transfer: u32,       // 0 = PQ (ST 2084), 1 = SDR gamma 2.4, 2 = HLG
    full_range: u32,
    bt2020: u32,         // primaries for the CIE scope: 1 = BT.2020, 0 = BT.709
    cll_base: u32,       // offset into cll for this frame (in u32s)
    sdr_white: f32,
    // YCbCr -> R'G'B' coefficients
    cr_r: f32,
    cb_g: f32,
    cr_g: f32,
    cb_b: f32,
    // linear RGB -> luminance weights
    kr: f32,
    kg: f32,
    kb: f32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> frame: array<u32>;
@group(0) @binding(2) var<storage, read_write> wg_stats: array<f32>;
@group(0) @binding(3) var<storage, read_write> scopes: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> cll: array<atomic<u32>>;

var<workgroup> s_max_rgb: array<f32, 256>;
var<workgroup> s_sum_rgb: array<f32, 256>;
var<workgroup> s_max_y: array<f32, 256>;
var<workgroup> s_sum_y: array<f32, 256>;
// Histogram is pre-aggregated per workgroup: only 4 x 256 bins, so global
// atomics would contend heavily.
var<workgroup> s_hist: array<atomic<u32>, 1024>;
var<workgroup> s_cll: array<atomic<u32>, 1024>;

fn sample(i: u32) -> f32 {
    let w = frame[i >> 1u];
    return f32((w >> ((i & 1u) * 16u)) & 0xffffu);
}

/// Bilinearly interpolated, normalized chroma (Cb, Cr) for luma pixel (x, y).
/// 4:2:0 chroma is assumed left-sited (MPEG-2 / HEVC default): co-sited with
/// even luma columns horizontally, halfway between luma rows vertically.
/// Interpolating matters: nearest-sample chroma next to a bright edge
/// produces R'G'B' overshoots that read as thousands of nits.
fn chroma(x: u32, y: u32) -> vec2<f32> {
    let cw = p.width / 2u;
    let ch = p.height / 2u;
    let cx = f32(x) * 0.5;
    let cy = max(f32(y) * 0.5 - 0.25, 0.0);
    let x0 = min(u32(floor(cx)), cw - 1u);
    let y0 = min(u32(floor(cy)), ch - 1u);
    let x1 = min(x0 + 1u, cw - 1u);
    let y1 = min(y0 + 1u, ch - 1u);
    let fx = cx - floor(cx);
    let fy = cy - floor(cy);
    let i00 = y0 * cw + x0;
    let i10 = y0 * cw + x1;
    let i01 = y1 * cw + x0;
    let i11 = y1 * cw + x1;
    let u = mix(mix(sample(p.u_off + i00), sample(p.u_off + i10), fx), mix(sample(p.u_off + i01), sample(p.u_off + i11), fx), fy);
    let v = mix(mix(sample(p.v_off + i00), sample(p.v_off + i10), fx), mix(sample(p.v_off + i01), sample(p.v_off + i11), fx), fy);
    let scale = select(896.0, 1023.0, p.full_range == 1u);
    return vec2<f32>((u - 512.0) / scale, (v - 512.0) / scale);
}

fn pq_eotf(e: f32) -> f32 {
    let m1 = 0.1593017578125;
    let m2 = 78.84375;
    let c1 = 0.8359375;
    let c2 = 18.8515625;
    let c3 = 18.6875;
    let ep = pow(clamp(e, 0.0, 1.0), 1.0 / m2);
    return 10000.0 * pow(max(ep - c1, 0.0) / (c2 - c3 * ep), 1.0 / m1);
}

fn hlg_inverse_oetf(e: f32) -> f32 {
    let a = 0.17883277;
    let b = 0.28466892;
    let c = 0.55991073;
    let x = clamp(e, 0.0, 1.0);
    if (x <= 0.5) {
        return x * x / 3.0;
    }
    return (exp((x - c) / a) + b) / 12.0;
}

/// Converts one pixel's non-linear R'G'B' to linear RGB in nits.
fn to_nits(rgb_p: vec3<f32>) -> vec3<f32> {
    if (p.transfer == 0u) {
        return vec3<f32>(pq_eotf(rgb_p.r), pq_eotf(rgb_p.g), pq_eotf(rgb_p.b));
    }
    if (p.transfer == 2u) {
        // HLG with the BT.2100 reference OOTF for a 1000-nit display (gamma 1.2).
        let s = vec3<f32>(hlg_inverse_oetf(rgb_p.r), hlg_inverse_oetf(rgb_p.g), hlg_inverse_oetf(rgb_p.b));
        let ys = p.kr * s.r + p.kg * s.g + p.kb * s.b;
        return 1000.0 * pow(max(ys, 1e-6), 0.2) * s;
    }
    let c = clamp(rgb_p, vec3<f32>(0.0), vec3<f32>(1.0));
    return p.sdr_white * pow(c, vec3<f32>(2.4));
}

/// log10(n + 1) / log10(10001) scaled to `bins`: 0 nits -> bin 0, 10000 -> last bin.
fn log_bin(nits: f32, bins: u32) -> u32 {
    let t = clamp(log2(nits + 1.0) / log2(10001.0), 0.0, 1.0);
    return u32(t * f32(bins - 1u) + 0.5);
}

fn add_wave(channel: u32, col: u32, nits: f32) {
    let idx = p.scope_base + (channel * p.wave_rows + log_bin(nits, p.wave_rows)) * p.wave_cols + col;
    atomicAdd(&scopes[idx], 1u);
}

/// CIE 1931 xy chromaticity of linear RGB in the stream's primaries.
fn chromaticity(n: vec3<f32>) -> vec2<f32> {
    var xyz: vec3<f32>;
    if (p.bt2020 == 1u) {
        xyz = vec3<f32>(
            0.636958 * n.r + 0.144617 * n.g + 0.168881 * n.b,
            0.262700 * n.r + 0.677998 * n.g + 0.059302 * n.b,
            0.028073 * n.g + 1.060985 * n.b);
    } else {
        xyz = vec3<f32>(
            0.4124 * n.r + 0.3576 * n.g + 0.1805 * n.b,
            0.2126 * n.r + 0.7152 * n.g + 0.0722 * n.b,
            0.0193 * n.r + 0.1192 * n.g + 0.9505 * n.b);
    }
    let sum = xyz.x + xyz.y + xyz.z;
    return vec2<f32>(xyz.x / sum, xyz.y / sum);
}

fn xy_bin(v: f32, lo: f32, hi: f32) -> u32 {
    return min(u32(clamp((v - lo) / (hi - lo), 0.0, 1.0) * f32(XY_BINS)), XY_BINS - 1u);
}

@compute @workgroup_size(16, 16)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
) {
    let do_wave = (p.scope_flags & SCOPE_WAVE) != 0u;
    let do_hist = (p.scope_flags & SCOPE_HIST) != 0u;
    let do_cie = (p.scope_flags & SCOPE_CIE) != 0u;
    let do_vec = (p.scope_flags & SCOPE_VEC) != 0u;
    for (var i = 0u; i < 4u; i = i + 1u) {
        atomicStore(&s_cll[li * 4u + i], 0u);
        if (do_hist) {
            atomicStore(&s_hist[li * 4u + i], 0u);
        }
    }
    workgroupBarrier();

    var max_rgb = 0.0;
    var sum_rgb = 0.0;
    var max_y = 0.0;
    var sum_y = 0.0;

    let cw = p.width / 2u;
    let ch = p.height / 2u;
    if (gid.x < cw && gid.y < ch) {
        for (var k = 0u; k < 4u; k = k + 1u) {
            let x = gid.x * 2u + (k & 1u);
            let y = gid.y * 2u + (k >> 1u);
            let c = chroma(x, y);
            var luma = sample(y * p.width + x);
            if (p.full_range == 1u) {
                luma = luma / 1023.0;
            } else {
                luma = (luma - 64.0) / 876.0;
            }
            let rgb_p = vec3<f32>(luma + p.cr_r * c.y, luma - p.cb_g * c.x - p.cr_g * c.y, luma + p.cb_b * c.x);
            let n = to_nits(rgb_p);
            let m = max(n.r, max(n.g, n.b));
            let lum = p.kr * n.r + p.kg * n.g + p.kb * n.b;
            atomicAdd(&s_cll[log_bin(m, CLL_BINS)], 1u);
            max_rgb = max(max_rgb, m);
            sum_rgb = sum_rgb + m;
            max_y = max(max_y, lum);
            sum_y = sum_y + lum;
            if (do_wave) {
                let col = x * p.wave_cols / p.width;
                add_wave(0u, col, n.r);
                add_wave(1u, col, n.g);
                add_wave(2u, col, n.b);
            }
            if (do_hist) {
                atomicAdd(&s_hist[log_bin(lum, HIST_BINS)], 1u);
                atomicAdd(&s_hist[HIST_BINS + log_bin(n.r, HIST_BINS)], 1u);
                atomicAdd(&s_hist[2u * HIST_BINS + log_bin(n.g, HIST_BINS)], 1u);
                atomicAdd(&s_hist[3u * HIST_BINS + log_bin(n.b, HIST_BINS)], 1u);
            }
            // 2D scopes sample one pixel per 2x2 block: plenty for a 256x256 plot,
            // and a quarter of the global atomics.
            if (k == 0u) {
                if (do_cie && lum > 0.05) {
                    let xy = chromaticity(n);
                    let bin = xy_bin(xy.y, 0.0, 0.9) * XY_BINS + xy_bin(xy.x, 0.0, 0.8);
                    atomicAdd(&scopes[p.scope_base + p.cie_off + bin], 1u);
                }
                if (do_vec) {
                    let bin = xy_bin(c.y, -0.5, 0.5) * XY_BINS + xy_bin(c.x, -0.5, 0.5);
                    atomicAdd(&scopes[p.scope_base + p.vec_off + bin], 1u);
                }
            }
        }
    }

    s_max_rgb[li] = max_rgb;
    s_sum_rgb[li] = sum_rgb;
    s_max_y[li] = max_y;
    s_sum_y[li] = sum_y;
    workgroupBarrier();
    for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
        if (li < stride) {
            s_max_rgb[li] = max(s_max_rgb[li], s_max_rgb[li + stride]);
            s_sum_rgb[li] = s_sum_rgb[li] + s_sum_rgb[li + stride];
            s_max_y[li] = max(s_max_y[li], s_max_y[li + stride]);
            s_sum_y[li] = s_sum_y[li] + s_sum_y[li + stride];
        }
        workgroupBarrier();
    }
    if (li == 0u) {
        let o = p.wg_base + (wid.y * nwg.x + wid.x) * 4u;
        wg_stats[o] = s_max_rgb[0];
        wg_stats[o + 1u] = s_sum_rgb[0];
        wg_stats[o + 2u] = s_max_y[0];
        wg_stats[o + 3u] = s_sum_y[0];
    }
    for (var i = 0u; i < 4u; i = i + 1u) {
        let bin = li * 4u + i;
        let v = atomicLoad(&s_cll[bin]);
        if (v > 0u) {
            atomicAdd(&cll[p.cll_base + bin], v);
        }
    }
    if (do_hist) {
        for (var i = 0u; i < 4u; i = i + 1u) {
            let bin = li * 4u + i;
            let v = atomicLoad(&s_hist[bin]);
            if (v > 0u) {
                atomicAdd(&scopes[p.scope_base + p.hist_off + bin], v);
            }
        }
    }
}
