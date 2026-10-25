// Blends the overlay (premultiplied RGBA8) into a 10-bit 4:2:0 frame in place.
// The frame is P010: a Y plane, then interleaved U/V, each 10-bit value in the
// top bits of a u16, two u16 per u32 word.
//
// One thread covers a 4x2 block of luma pixels: two luma words per row and two
// U/V words, so threads never share a word. Only
// blocks touching the overlay rectangle are dispatched.

struct Params {
    width: u32,          // frame width, a multiple of 4
    height: u32,         // frame height, a multiple of 2
    uv_off: u32,         // index (in u16 samples) of the interleaved U/V plane
    _pad3: u32,
    ox: u32,             // overlay rectangle in frame pixels
    oy: u32,
    ow: u32,
    oh: u32,
    block_x: u32,        // first dispatched block
    block_y: u32,
    full_range: u32,
    _pad0: u32,
    kr: f32,             // Y'CbCr matrix of the frame
    kb: f32,
    _pad1: f32,
    _pad2: f32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> frame: array<u32>;
@group(0) @binding(2) var<storage, read> overlay: array<u32>;
// Overlay code value (straight, 0..255) -> code value in the frame's transfer (0..255).
@group(0) @binding(3) var<storage, read> lut: array<u32, 256>;

/// Overlay pixel at frame (x, y) as 10-bit Y, Cb, Cr codes and alpha (0..1).
fn overlay_pixel(x: u32, y: u32) -> vec4<f32> {
    if (x < p.ox || y < p.oy || x >= p.ox + p.ow || y >= p.oy + p.oh) {
        return vec4<f32>(0.0);
    }
    let px = overlay[(y - p.oy) * p.ow + (x - p.ox)];
    let a8 = px >> 24u;
    if (a8 == 0u) {
        return vec4<f32>(0.0);
    }
    let a = f32(a8) / 255.0;
    // Straight colour through the transfer LUT, as normalised R'G'B'.
    let straight = vec3<f32>(f32(px & 0xffu), f32((px >> 8u) & 0xffu), f32((px >> 16u) & 0xffu)) / a;
    let idx = vec3<u32>(min(vec3<u32>(round(straight)), vec3<u32>(255u)));
    let rgb = vec3<f32>(f32(lut[idx.x]), f32(lut[idx.y]), f32(lut[idx.z])) / 255.0;
    let kg = 1.0 - p.kr - p.kb;
    let yp = p.kr * rgb.r + kg * rgb.g + p.kb * rgb.b;
    let cb = (rgb.b - yp) / (2.0 * (1.0 - p.kb));
    let cr = (rgb.r - yp) / (2.0 * (1.0 - p.kr));
    if (p.full_range == 1u) {
        return vec4<f32>(yp * 1023.0, 512.0 + cb * 1023.0, 512.0 + cr * 1023.0, a);
    }
    return vec4<f32>(64.0 + yp * 876.0, 512.0 + cb * 896.0, 512.0 + cr * 896.0, a);
}

fn to_code(v: f32) -> u32 {
    return u32(clamp(round(v), 0.0, 1023.0));
}

/// Blends two packed luma samples (pixels x, x+1) with their overlay pixels.
fn blend_word(word: u32, a: vec4<f32>, b: vec4<f32>) -> u32 {
    let lo = f32((word & 0xffffu) >> 6u);
    let hi = f32(word >> 22u);
    return (to_code(mix(lo, a.x, a.w)) << 6u) | (to_code(mix(hi, b.x, b.w)) << 22u);
}

/// Blends one chroma sample with the 2x2 overlay pixels it covers (box filter on
/// premultiplied chroma). `c` selects Cb (1) or Cr (2).
fn blend_chroma(dst: f32, q: array<vec4<f32>, 4>, c: u32) -> f32 {
    var sum_a = 0.0;
    var sum_ac = 0.0;
    for (var i = 0u; i < 4u; i = i + 1u) {
        sum_a = sum_a + q[i].w;
        sum_ac = sum_ac + q[i].w * q[i][c];
    }
    return sum_ac / 4.0 + (1.0 - sum_a / 4.0) * dst;
}

/// Blends one packed U/V pair with the 2x2 overlay pixels it covers.
fn blend_uv(word: u32, q: array<vec4<f32>, 4>) -> u32 {
    let u = f32((word & 0xffffu) >> 6u);
    let v = f32(word >> 22u);
    return (to_code(blend_chroma(u, q, 1u)) << 6u) | (to_code(blend_chroma(v, q, 2u)) << 22u);
}

@compute @workgroup_size(16, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let bx = p.block_x + gid.x;
    let by = p.block_y + gid.y;
    let x0 = bx * 4u;
    let y0 = by * 2u;
    if (x0 >= p.width || y0 >= p.height || x0 >= p.ox + p.ow || y0 >= p.oy + p.oh) {
        return;
    }
    // px[row * 4 + col]
    var px: array<vec4<f32>, 8>;
    var any = false;
    for (var i = 0u; i < 8u; i = i + 1u) {
        px[i] = overlay_pixel(x0 + (i & 3u), y0 + (i >> 2u));
        any = any || px[i].w > 0.0;
    }
    if (!any) {
        return;
    }
    for (var row = 0u; row < 2u; row = row + 1u) {
        let w = ((y0 + row) * p.width + x0) / 2u;
        frame[w] = blend_word(frame[w], px[row * 4u], px[row * 4u + 1u]);
        frame[w + 1u] = blend_word(frame[w + 1u], px[row * 4u + 2u], px[row * 4u + 3u]);
    }
    // The two chroma samples of this block, each covering a 2x2 group.
    let left = array<vec4<f32>, 4>(px[0], px[1], px[4], px[5]);
    let right = array<vec4<f32>, 4>(px[2], px[3], px[6], px[7]);
    let w = p.uv_off / 2u + by * (p.width / 2u) + bx * 2u;
    frame[w] = blend_uv(frame[w], left);
    frame[w + 1u] = blend_uv(frame[w + 1u], right);
}
