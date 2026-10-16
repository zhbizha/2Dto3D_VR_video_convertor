namespace VRify.Core;

/// <summary>HLSL compute source compiled at runtime (cs_5_0).</summary>
internal static class GpuShaders
{
    public const string Source = """
cbuffer CB : register(b0)
{
    uint  cW;
    uint  cH;
    float cMaxShift;
    float cZTol;
    float cAStat;
    float cAMot;
    float cEps;
    uint  cRadius;
    uint  cFirst;
    uint  cEyeW;
    uint  cEyeH;
    float cKx;
    float cKy;
    uint  cPad0;
    uint  cPad1;
    uint  cPad2;
};

#define DISP_SCALE 64.0

// ---- RGB24 byte-address helpers (3-byte pixels, Buffer<uint> = byteOffset>>2) ----
uint3 LoadRGB(Buffer<uint> b, uint byteOff)
{
    uint a = byteOff >> 2;
    uint k = byteOff & 3u;
    uint v0 = b[a];
    uint v1 = b[a + 1];
    uint r = (v0 >> (8u * k)) & 0xFFu;
    uint g = (k == 3u) ? (v1 & 0xFFu) : ((v0 >> (8u * k + 8u)) & 0xFFu);
    uint bb = (k <= 1u) ? ((v0 >> (8u * k + 16u)) & 0xFFu)
            : (k == 2u) ? (v1 & 0xFFu) : ((v1 >> 8u) & 0xFFu);
    return uint3(r, g, bb);
}

float3 LoadRGBf(Buffer<uint> b, uint byteOff)
{
    return (float3)LoadRGB(b, byteOff);
}

float LumaOf(float3 c) // c in 0..255
{
    return (0.299 * c.r + 0.587 * c.g + 0.114 * c.b) / 255.0;
}

// packed RGBA: a != 0 means the destination pixel received a splat (not a hole).
bool IsKnown(uint v) { return (v & 255u) != 0u; }

// ============================================================
// Temporal depth smooth guided by RGB change
StructuredBuffer<float> gRaw : register(t0);
StructuredBuffer<float> gHist : register(t1);
Buffer<uint> gRgb : register(t2);
Buffer<uint> gPrevRgb : register(t3);
RWStructuredBuffer<float> gOut : register(u0);

[numthreads(8, 8, 1)]
void Smooth(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= cW || id.y >= cH) return;
    uint i = id.y * cW + id.x;
    float d = gRaw[i];
    if (cFirst != 0) { gOut[i] = d; return; }
    float prev = gHist[i];
    uint p = i * 3u;
    float3 c = LoadRGBf(gRgb, p);
    float3 pc = LoadRGBf(gPrevRgb, p);
    float chg = max(abs(c.r - pc.r), max(abs(c.g - pc.g), abs(c.b - pc.b)));
    float st = chg <= 6.0 ? 1.0 : (chg >= 24.0 ? 0.0 : 1.0 - (chg - 6.0) / 18.0);
    float alpha = cAMot + (cAStat - cAMot) * st;
    gOut[i] = prev + (d - prev) * alpha;
}

// ============================================================
// Build guide stats: (I, I*I, P, I*P)
StructuredBuffer<float> gPIn : register(t0);
Buffer<uint> gRgbG : register(t1);
RWStructuredBuffer<float4> gA : register(u0);

[numthreads(8, 8, 1)]
void GuideBuild(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= cW || id.y >= cH) return;
    uint i = id.y * cW + id.x;
    float P = gPIn[i];
    float I = LumaOf(LoadRGBf(gRgbG, i * 3u));
    gA[i] = float4(I, I * I, P, I * P);
}

// ============================================================
// Box mean, horizontal: group = 64-wide row segment
StructuredBuffer<float4> gBHin : register(t0);
RWStructuredBuffer<float4> gBHout : register(u0);
groupshared float4 gTile[96];

[numthreads(64, 1, 1)]
void BoxH(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    uint y = gid.y;
    uint segX = gid.x * 64u;
    uint R = cRadius;
    uint S = 64u + 2u * R;
    for (uint j = tid.x; j < S; j += 64u)
    {
        int xs = (int)(segX + j) - (int)R;
        xs = clamp(xs, 0, (int)cW - 1);
        gTile[j] = gBHin[y * cW + (uint)xs];
    }
    GroupMemoryBarrierWithGroupSync();
    uint x = segX + tid.x;
    if (x >= cW) return;
    int x0 = max((int)x - (int)R, 0);
    int x1 = min((int)x + (int)R, (int)cW - 1);
    uint t0 = (uint)(x0 - (int)segX + (int)R);
    uint t1 = (uint)(x1 - (int)segX + (int)R);
    float4 s = 0.0;
    for (uint j2 = t0; j2 <= t1; j2++) s += gTile[j2];
    gBHout[y * cW + x] = s / (float)(x1 - x0 + 1);
}

// ============================================================
// Box mean, vertical: group = 64-tall column segment
StructuredBuffer<float4> gBVin : register(t0);
RWStructuredBuffer<float4> gBVout : register(u0);

[numthreads(1, 64, 1)]
void BoxV(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    uint x = gid.x;
    uint segY = gid.y * 64u;
    uint R = cRadius;
    uint S = 64u + 2u * R;
    for (uint j = tid.y; j < S; j += 64u)
    {
        int ys = (int)(segY + j) - (int)R;
        ys = clamp(ys, 0, (int)cH - 1);
        gTile[j] = gBVin[(uint)ys * cW + x];
    }
    GroupMemoryBarrierWithGroupSync();
    uint y = segY + tid.y;
    if (y >= cH) return;
    int y0 = max((int)y - (int)R, 0);
    int y1 = min((int)y + (int)R, (int)cH - 1);
    uint t0 = (uint)(y0 - (int)segY + (int)R);
    uint t1 = (uint)(y1 - (int)segY + (int)R);
    float4 s = 0.0;
    for (uint j2 = t0; j2 <= t1; j2++) s += gTile[j2];
    gBVout[y * cW + x] = s / (float)(y1 - y0 + 1);
}

// ============================================================
// Guided filter: A,B coefficients from box means of (I,I2,P,IP)
// Input: mean of (I, I2, P, IP) as float4; output (a, b, 0, 0)
RWStructuredBuffer<float4> gMeans : register(u0);

[numthreads(8, 8, 1)]
void ABPass(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= cW || id.y >= cH) return;
    uint i = id.y * cW + id.x;
    float4 m = gMeans[i];
    float var = m.y - m.x * m.x;
    float cov = m.w - m.x * m.z;
    float a = cov / (var + cEps);
    float b = m.z - a * m.x;
    gMeans[i] = float4(a, b, 0, 0);
}

// ============================================================
// Final guided output: q = ma*I + mb  (reads gMeans=ma,mb + gA.x=I)
StructuredBuffer<float4> gMB : register(t0);
StructuredBuffer<float4> gA2 : register(t1);
RWStructuredBuffer<float> gQ : register(u0);

[numthreads(8, 8, 1)]
void GuidedFinal(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= cW || id.y >= cH) return;
    uint i = id.y * cW + id.x;
    float4 ab = gMB[i];
    float I = gA2[i].x;
    float q = ab.x * I + ab.y;
    gQ[i] = clamp(q, 0.0, 1.0);
}

// ============================================================
// Median 3x3 on scalar buffer
StructuredBuffer<float> gMedIn : register(t0);
RWStructuredBuffer<float> gMedOut : register(u0);

[numthreads(8, 8, 1)]
void Median3(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= cW || id.y >= cH) return;
    int x = (int)id.x, y = (int)id.y;
    float v[9];
    int n = 0;
    for (int dy = -1; dy <= 1; dy++)
        for (int dx = -1; dx <= 1; dx++)
        {
            int sx = clamp(x + dx, 0, (int)cW - 1);
            int sy = clamp(y + dy, 0, (int)cH - 1);
            v[n++] = gMedIn[(uint)sy * cW + (uint)sx];
        }
    // insertion sort of 9
    for (int i = 1; i < 9; i++)
    {
        float key = v[i];
        int j = i - 1;
        while (j >= 0 && v[j] > key) { v[j + 1] = v[j]; j--; }
        v[j + 1] = key;
    }
    gMedOut[(uint)y * cW + (uint)x] = v[4];
}

// ============================================================
// Bilinear upsample of depth (low-res -> full-res)
StructuredBuffer<float> gUpIn : register(t0);
RWStructuredBuffer<float> gUpOut : register(u0);

[numthreads(8, 8, 1)]
void Upsample(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= cW || id.y >= cH) return; // cW,cH = destination size
    uint i = id.y * cW + id.x;
    // cKx, cKy = source size
    float fx = ((float)id.x + 0.5) * cKx / (float)cW - 0.5;
    float fy = ((float)id.y + 0.5) * cKy / (float)cH - 0.5;
    int x0 = (int)floor(fx), y0 = (int)floor(fy);
    float tx = fx - (float)x0, ty = fy - (float)y0;
    int sw = (int)cKx, sh = (int)cKy;
    int x1 = clamp(x0 + 1, 0, sw - 1), y1 = clamp(y0 + 1, 0, sh - 1);
    x0 = clamp(x0, 0, sw - 1); y0 = clamp(y0, 0, sh - 1);
    float a = gUpIn[(uint)y0 * (uint)sw + (uint)x0];
    float b = gUpIn[(uint)y0 * (uint)sw + (uint)x1];
    float c = gUpIn[(uint)y1 * (uint)sw + (uint)x0];
    float d = gUpIn[(uint)y1 * (uint)sw + (uint)x1];
    float top = a + (b - a) * tx;
    float bot = c + (d - c) * tx;
    gUpOut[i] = top + (bot - top) * ty;
}

// ============================================================
// Gated ramp collapse - mirrors DepthEstimator.CollapseRamps; keep COL_KS/COL_KB/COL_GAIN
// in sync with DepthEstimator.FullResKernelS/B/Gain.
// stepness = range(+-KS) / range(+-KB) is ~1 at a genuine discontinuity and ~KS/KB on a
// uniform slope, so the local contrast stretch is blended in only where the structure is
// actually a step, leaving real depth slopes bit-exact.
StructuredBuffer<float> gColIn : register(t0);
RWStructuredBuffer<float> gColOut : register(u0);

#define COL_KS 12
#define COL_KB 26
#define COL_GAIN 5.0

[numthreads(64, 1, 1)]
void CollapseRamps(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= cW || id.y >= cH) return;
    uint row = id.y * cW;
    int x = (int)id.x;

    float loS = 1e30, hiS = -1e30, loB = 1e30, hiB = -1e30;
    int s0 = max(0, x - COL_KS), s1 = min((int)cW - 1, x + COL_KS);
    int b0 = max(0, x - COL_KB), b1 = min((int)cW - 1, x + COL_KB);
    for (int k = s0; k <= s1; k++) { float v = gColIn[row + (uint)k]; loS = min(loS, v); hiS = max(hiS, v); }
    for (int k = b0; k <= b1; k++) { float v = gColIn[row + (uint)k]; loB = min(loB, v); hiB = max(hiB, v); }

    float rS = hiS - loS, rB = hiB - loB;
    float stepness = rB > 1e-6 ? rS / rB : 0.0;
    float t = clamp((stepness - 0.5) / 0.3, 0.0, 1.0);
    float wt = t * t * (3.0 - 2.0 * t);

    float v0 = gColIn[row + (uint)x];
    if (wt <= 0.0 || rS < 0.02) { gColOut[row + (uint)x] = v0; return; }

    float tt = (v0 - loS) / rS;
    float tc = clamp((tt - 0.5) * COL_GAIN + 0.5, 0.0, 1.0);
    gColOut[row + (uint)x] = v0 + (loS + tc * rS - v0) * wt;
}

// ============================================================
// Hard-step pass: mirrors DepthEstimator.HardStepPass. After the gated ramp
// collapse, walk each row and wherever a small inner window still has a sharp
// step inside a wider window, snap to whichever pole is closer. This turns the
// last 4-8 px of bilinear-upsample AA into a true step on the warp input,
// killing the "clothing fragment over disocclusion" stair on the right eye.
// Keep HardStepKD/KB/StepMin/Slope in sync with the DepthEstimator defaults.
StructuredBuffer<float> gHsIn : register(t0);
RWStructuredBuffer<float> gHsOut : register(u0);

#define HARDSTEP_KD 3
#define HARDSTEP_KB 10
#define HARDSTEP_STEPMIN 0.75
#define HARDSTEP_SLOPE 0.12

[numthreads(64, 1, 1)]
void HardStepPass(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= cW || id.y >= cH) return;
    uint row = id.y * cW;
    int x = (int)id.x;
    int s0 = max(0, x - HARDSTEP_KD), s1 = min((int)cW - 1, x + HARDSTEP_KD);
    int b0 = max(0, x - HARDSTEP_KB), b1 = min((int)cW - 1, x + HARDSTEP_KB);
    float mnS = 1e30, mxS = -1e30, mnB = 1e30, mxB = -1e30;
    for (int k = s0; k <= s1; k++) { float v = gHsIn[row + (uint)k]; mnS = min(mnS, v); mxS = max(mxS, v); }
    if (mxS - mnS <= HARDSTEP_SLOPE) { gHsOut[row + (uint)x] = gHsIn[row + (uint)x]; return; }
    for (int k = b0; k <= b1; k++) { float v = gHsIn[row + (uint)k]; mnB = min(mnB, v); mxB = max(mxB, v); }
    float rB = mxB - mnB;
    if (rB < 1e-6) { gHsOut[row + (uint)x] = gHsIn[row + (uint)x]; return; }
    float stepness = (mxS - mnS) / rB;
    if (stepness < HARDSTEP_STEPMIN) { gHsOut[row + (uint)x] = gHsIn[row + (uint)x]; return; }
    float mid = (mnB + mxB) * 0.5;
    int nLow = 0, nHigh = 0;
    for (int k = b0; k <= b1; k++) { if (gHsIn[row + (uint)k] < mid) nLow++; else nHigh++; }
    if (min(nLow, nHigh) < 2) { gHsOut[row + (uint)x] = gHsIn[row + (uint)x]; return; }
    gHsOut[row + (uint)x] = (mid >= gHsIn[row + (uint)x]) ? mxB : mnB;
}

// ============================================================
// Forward-warp GATHER: for destination xd scan sources xs in [xd, xd+maxShift+2),
// replicate CPU splat semantics (bilinear weight + z-test replace/blend).
StructuredBuffer<float> gWarpDepth : register(t0);
Buffer<uint> gWarpRgb : register(t1);
RWStructuredBuffer<uint> gWarpOut : register(u0); // packed RGBA (a=255 known, 0 hole)
RWStructuredBuffer<float> gWarpDisp : register(u1); // winner disparity (=-1 if hole)

[numthreads(8, 8, 1)]
void WarpGather(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= cW || id.y >= cH) return;
    uint xd = id.x;
    uint row = id.y * cW;
    float maxShift = cMaxShift;
    float zTol = cZTol;
    float bestD = -1.0;
    float3 acc = 0;
    float wsum = 0;
    uint xsEnd = min(cW, xd + (uint)ceil(maxShift) + 2u);
    for (uint xs = xd; xs < xsEnd; xs++)
    {
        float disp = gWarpDepth[row + xs] * maxShift;
        float xf = (float)xs - disp;
        if (xf < -1.0 || xf > (float)cW) continue;
        float x0f = trunc(xf);
        float t = xf - x0f;
        float w;
        if ((float)xd == x0f) w = 1.0 - t;
        else if ((float)xd == x0f + 1.0 && t > 0.0) w = t;
        else continue;
        if (w <= 0.0) continue;

        if (disp > bestD + zTol)
        {   // clearly nearer (or empty) -> replace
            bestD = disp;
            wsum = w;
            acc = LoadRGBf(gWarpRgb, (row + xs) * 3u) * w;
        }
        else if (disp >= bestD - zTol)
        {   // same surface -> blend
            wsum += w;
            acc += LoadRGBf(gWarpRgb, (row + xs) * 3u) * w;
        }
        // else: occluded by nearer geometry -> drop
    }
    uint px = 0;
    if (wsum > 1e-6)
    {
        float3 c = acc / wsum;
        uint r = (uint)clamp(c.r, 0.0, 255.0);
        uint g = (uint)clamp(c.g, 0.0, 255.0);
        uint b = (uint)clamp(c.b, 0.0, 255.0);
        px = (r << 24) | (g << 16) | (b << 8) | 255u;
    }
    gWarpOut[row + xd] = px;
    gWarpDisp[row + xd] = (wsum > 1e-6) ? bestD : -1.0;
}

// ============================================================
// Background-mirror disocclusion fill (mirrors StereoSynthesizer.FillHolesFromBackground).
// A hole in a horizontal-disparity warp always sits between a near region and a far
// region, and the far one is the revealed background. Take that side (smallest
// disparity), map the run into its source domain and mirror the real background pixels
// about the run boundary - the hole then continues actual background texture instead of
// averaging the foreground in, which is what the push-pull blur did and what read as a
// halo / rubber band on silhouettes.
StructuredBuffer<uint>   gBgIn   : register(t0); // packed RGBA (a!=0 = known)
StructuredBuffer<float>  gBgDisp : register(t1); // winner disparity (-1 = hole)
Buffer<uint>             gBgRgb  : register(t2); // source RGB24
RWStructuredBuffer<uint> gBgOut  : register(u0);

[numthreads(8, 8, 1)]
void BgFill(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= cW || id.y >= cH) return;
    uint xd = id.x;
    uint row = id.y * cW;
    uint v = gBgIn[row + xd];
    if (IsKnown(v)) { gBgOut[row + xd] = v; return; }

    int maxScan = (int)ceil(cMaxShift) + 2;
    int a = -1; float dl = 0.0;
    for (int k = 1; k <= maxScan; k++)
    {
        int x = (int)xd - k;
        if (x < 0) break;
        if (IsKnown(gBgIn[row + (uint)x])) { a = x; dl = gBgDisp[row + (uint)x]; break; }
    }
    int b = -1; float dr = 0.0;
    for (int k = 1; k <= maxScan; k++)
    {
        int x = (int)xd + k;
        if (x >= (int)cW) break;
        if (IsKnown(gBgIn[row + (uint)x])) { b = x; dr = gBgDisp[row + (uint)x]; break; }
    }

    float anchor, dispFar;
    if (a >= 0 && (b < 0 || dl <= dr)) { anchor = (float)a; dispFar = dl; }
    else if (b >= 0)                   { anchor = (float)b; dispFar = dr; }
    else { gBgOut[row + xd] = 0u; return; }   // whole row is a hole -> leave for push-pull

    float xs = 2.0 * anchor + dispFar - (float)xd;   // mirror in the background's source domain
    if (xs < 0.0) xs = -xs;
    if (xs > (float)(cW - 1u)) xs = 2.0 * (float)(cW - 1u) - xs;
    xs = clamp(xs, 0.0, (float)(cW - 1u));

    uint s0 = (uint)xs;
    uint s1 = min(s0 + 1u, cW - 1u);
    float t = xs - (float)s0;
    float3 c0 = LoadRGBf(gBgRgb, (row + s0) * 3u);
    float3 c1 = LoadRGBf(gBgRgb, (row + s1) * 3u);
    float3 c = c0 + (c1 - c0) * t;
    uint r = (uint)clamp(c.r, 0.0, 255.0);
    uint g = (uint)clamp(c.g, 0.0, 255.0);
    uint bl = (uint)clamp(c.b, 0.0, 255.0);
    gBgOut[row + xd] = (r << 24) | (g << 16) | (bl << 8) | 255u;
}

// ============================================================
// Thin horizontal hole fill (runs <= 5 px), input->output ping-pong.
StructuredBuffer<uint> gThinIn : register(t0);
RWStructuredBuffer<uint> gThinOut : register(u0);

// packed RGBA: r<<24 | g<<16 | b<<8 | a
uint3 UnpackRgb(uint v) { return uint3((v >> 24) & 255u, (v >> 16) & 255u, (v >> 8) & 255u); }

[numthreads(8, 8, 1)]
void ThinFill(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= cW || id.y >= cH) return;
    uint xd = id.x;
    uint row = id.y * cW;
    uint v = gThinIn[row + xd];
    if (IsKnown(v)) { gThinOut[row + xd] = v; return; }

    const uint MAXR = 5u;
    // nearest known left within MAXR (or row boundary)
    uint dl = 0; bool foundL = false, boundL = false;
    for (uint kl = 1u; kl <= MAXR; kl++)
    {
        if (xd < kl) { boundL = true; break; }
        if (IsKnown(gThinIn[row + xd - kl])) { dl = kl; foundL = true; break; }
    }
    // nearest known right within MAXR (or row boundary)
    uint dr = 0; bool foundR = false, boundR = false;
    for (uint kr = 1u; kr <= MAXR; kr++)
    {
        if (xd + kr >= cW) { boundR = true; break; }
        if (IsKnown(gThinIn[row + xd + kr])) { dr = kr; foundR = true; break; }
    }

    uint3 col = 0;
    bool fill = false;
    if (foundL && foundR)
    {
        uint runLen = dl + dr - 1u;
        if (runLen <= MAXR)
        {
            float t = (float)dl / (float)(dl + dr + 1u);
            float3 l = UnpackRgb(gThinIn[row + xd - dl]);
            float3 r = UnpackRgb(gThinIn[row + xd + dr]);
            col = (uint3)(l + (r - l) * t);
            fill = true;
        }
    }
    else if (foundL && boundR)
    {
        // run reaches the right edge: len = dl + (W-1-xd)
        uint runLen = dl + (cW - 1u - xd);
        if (runLen <= MAXR) { col = UnpackRgb(gThinIn[row + xd - dl]); fill = true; }
    }
    else if (foundR && boundL)
    {
        uint runLen = xd + dr;
        if (runLen <= MAXR) { col = UnpackRgb(gThinIn[row + xd + dr]); fill = true; }
    }

    gThinOut[row + xd] = fill
        ? ((col.r << 24) | (col.g << 16) | (col.b << 8) | 255u)
        : 0u;
}

// ============================================================
// Push-pull pyramid inpaint: init level0 from packed RGBA, then down/up passes.
StructuredBuffer<uint> gPyIn : register(t0);
RWStructuredBuffer<float4> gPyLevel0 : register(u0);

[numthreads(8, 8, 1)]
void PyInit(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= cW || id.y >= cH) return;
    uint i = id.y * cW + id.x;
    uint v = gPyIn[i];
    float known = (v & 255u) != 0u ? 1.0 : 0.0;
    float3 c = float3((v >> 24) & 255u, (v >> 16) & 255u, (v >> 8) & 255u);
    gPyLevel0[i] = float4(c * known, known);
}

// Down: coarse(x,y) averages 2x2 children of fine (clamped like CPU).
StructuredBuffer<float4> gPyFine : register(t0);
RWStructuredBuffer<float4> gPyCoarse : register(u0);

[numthreads(8, 8, 1)]
void PyDown(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= cW || id.y >= cH) return;   // cW,cH = COARSE dims; fine dims in cKx,cKy
    int cw = (int)cKx, ch = (int)cKy;
    int sx = min(cw - 1, (int)id.x * 2), sx2 = min(cw - 1, (int)id.x * 2 + 1);
    int sy = min(ch - 1, (int)id.y * 2), sy2 = min(ch - 1, (int)id.y * 2 + 1);
    float4 a = gPyFine[sy * cw + sx];
    float4 b = gPyFine[sy * cw + sx2];
    float4 c = gPyFine[sy2 * cw + sx];
    float4 d = gPyFine[sy2 * cw + sx2];
    float wsum = a.w + b.w + c.w + d.w;
    float4 o = 0;
    if (wsum > 0)
    {
        // children store premultiplied rgb, so plain weighted avg works
        o = (a + b + c + d) / wsum;
        o.w = 1;
    }
    gPyCoarse[id.y * cW + id.x] = o;
}

// Up: fine keeps own color if known, else bilinear estimate from coarse.
StructuredBuffer<float4> gPyFineU : register(t0);
StructuredBuffer<float4> gPyCoarseU : register(t1);
RWStructuredBuffer<float4> gPyFineOut : register(u0);

[numthreads(8, 8, 1)]
void PyUp(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= cW || id.y >= cH) return;   // cW,cH = FINE dims; coarse dims in cKx,cKy
    int cw = (int)cKx, ch = (int)cKy;
    int i = (int)id.y * (int)cW + (int)id.x;
    float4 own = gPyFineU[i];
    if (own.w > 0) { gPyFineOut[i] = own; return; }
    float fy = clamp(((float)id.y + 0.5f) * (float)ch / (float)cH - 0.5f, 0.0, (float)ch - 1.0f);
    float fx = clamp(((float)id.x + 0.5f) * (float)cw / (float)cW - 0.5f, 0.0, (float)cw - 1.0f);
    int y0 = (int)fy, y1 = min(y0 + 1, ch - 1);
    int x0 = (int)fx, x1 = min(x0 + 1, cw - 1);
    float tx = fx - (float)x0, ty = fy - (float)y0;
    float4 c00 = gPyCoarseU[y0 * cw + x0], c01 = gPyCoarseU[y0 * cw + x1];
    float4 c10 = gPyCoarseU[y1 * cw + x0], c11 = gPyCoarseU[y1 * cw + x1];
    float4 est = c00 * ((1 - tx) * (1 - ty)) + c01 * (tx * (1 - ty))
               + c10 * ((1 - tx) * ty)       + c11 * (tx * ty);
    est.w = 1;
    gPyFineOut[i] = est;
}

// Final: write filled pixels back into packed RGBA buffer (holes only).
StructuredBuffer<float4> gFinLevel0 : register(t0);
StructuredBuffer<uint> gFinThin : register(t1);
RWStructuredBuffer<uint> gFinOut : register(u0);

[numthreads(8, 8, 1)]
void PyFinalize(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= cW || id.y >= cH) return;
    uint i = id.y * cW + id.x;
    uint prev = gFinThin[i];
    if ((prev & 255u) != 0u) { gFinOut[i] = prev; return; } // known pixel: keep original
    float4 f = gFinLevel0[i];
    uint r = (uint)clamp(f.r, 0.0, 255.0);
    uint g = (uint)clamp(f.g, 0.0, 255.0);
    uint b = (uint)clamp(f.b, 0.0, 255.0);
    gFinOut[i] = (r << 24) | (g << 16) | (b << 8) | 255u;
}

// ============================================================
// Pack SBS: output RGBA8 (4 B/px, aligned uint stores - no byte races).
// cW,cH = src dims (left/right eye); cEyeW = dst frame width in pixels (2*W).
Buffer<uint> gPackRgb : register(t0);
StructuredBuffer<uint> gPackRight : register(t1);
RWStructuredBuffer<uint> gPackOut : register(u0);

[numthreads(8, 8, 1)]
void PackSbs(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= cW || id.y >= cH) return;
    uint3 lc = LoadRGB(gPackRgb, (id.y * cW + id.x) * 3u);
    uint rv = gPackRight[id.y * cW + id.x];
    uint3 rc = UnpackRgb(rv);
    uint rowBase = id.y * cEyeW + id.x;
    gPackOut[rowBase]            = lc.x | (lc.y << 8) | (lc.z << 16) | 0xFF000000u;
    gPackOut[rowBase + cW]       = rc.x | (rc.y << 8) | (rc.z << 16) | 0xFF000000u;
}

// ============================================================
// Equirect project one eye -> RGBA8. Two specialised entry points (left, right)
// hard-code which SRV slot is the colour source, so the CB never needs an "eye
// select" field (a previous cPad1-based branch silently degenerated to the same
// eye on both passes). cPad0 is still the per-pass dst x offset; cW,cH are the
// flat src dims; cEyeW,cEyeH are the equirect eye dims; cKx,cKy are 1/tanHalf
// and the matching vertical scale.
Buffer<uint> gEqSrc : register(t0);
StructuredBuffer<uint> gEqRight : register(t1);
StructuredBuffer<float> gEqTanLon : register(t2);
StructuredBuffer<float> gEqInvCos : register(t3);
StructuredBuffer<float> gEqTanLat : register(t4);
RWStructuredBuffer<uint> gEqOut : register(u0);

float3 EquirectSampleSrc(int x0, int x1, int y0, int y1, float w00, float w01, float w10, float w11)
{
    uint3 p00 = LoadRGB(gEqSrc, ((uint)y0 * (uint)cW + (uint)x0) * 3u);
    uint3 p01 = LoadRGB(gEqSrc, ((uint)y0 * (uint)cW + (uint)x1) * 3u);
    uint3 p10 = LoadRGB(gEqSrc, ((uint)y1 * (uint)cW + (uint)x0) * 3u);
    uint3 p11 = LoadRGB(gEqSrc, ((uint)y1 * (uint)cW + (uint)x1) * 3u);
    return (float3)p00 * w00 + (float3)p01 * w01 + (float3)p10 * w10 + (float3)p11 * w11;
}

float3 EquirectSampleRight(int x0, int x1, int y0, int y1, float w00, float w01, float w10, float w11)
{
    uint q00 = gEqRight[y0 * (int)cW + x0], q01 = gEqRight[y0 * (int)cW + x1];
    uint q10 = gEqRight[y1 * (int)cW + x0], q11 = gEqRight[y1 * (int)cW + x1];
    return (float3)UnpackRgb(q00) * w00 + (float3)UnpackRgb(q01) * w01
         + (float3)UnpackRgb(q10) * w10 + (float3)UnpackRgb(q11) * w11;
}

[numthreads(8, 8, 1)]
void EquirectProjectLeft(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= cEyeW || id.y >= cEyeH) return;
    float u = gEqTanLon[id.x] * cKx;
    float v = gEqTanLat[id.y] * gEqInvCos[id.x] * cKy;
    uint dstIndex = id.y * (2u * cEyeW) + cPad0 + id.x;

    if (u <= -1.0 || u >= 1.0 || v <= -1.0 || v >= 1.0)
    {
        gEqOut[dstIndex] = 0xFF000000u;   // outside screen -> black
        return;
    }
    float fx = clamp((u + 1.0) * 0.5 * (float)cW - 0.5, 0.0, (float)cW - 1.0);
    float fy = clamp((0.5 - v * 0.5) * (float)cH - 0.5, 0.0, (float)cH - 1.0);
    int x0 = (int)fx, y0 = (int)fy;
    int x1 = min(x0 + 1, (int)cW - 1), y1 = min(y0 + 1, (int)cH - 1);
    float tx = fx - (float)x0, ty = fy - (float)y0;
    float w00 = (1 - tx) * (1 - ty), w01 = tx * (1 - ty),
          w10 = (1 - tx) * ty,       w11 = tx * ty;

    float3 s = EquirectSampleSrc(x0, x1, y0, y1, w00, w01, w10, w11);
    uint3 c = (uint3)clamp(s, 0.0, 255.0);
    gEqOut[dstIndex] = c.x | (c.y << 8) | (c.z << 16) | 0xFF000000u;
}

[numthreads(8, 8, 1)]
void EquirectProjectRight(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= cEyeW || id.y >= cEyeH) return;
    float u = gEqTanLon[id.x] * cKx;
    float v = gEqTanLat[id.y] * gEqInvCos[id.x] * cKy;
    uint dstIndex = id.y * (2u * cEyeW) + cPad0 + id.x;

    if (u <= -1.0 || u >= 1.0 || v <= -1.0 || v >= 1.0)
    {
        gEqOut[dstIndex] = 0xFF000000u;   // outside screen -> black
        return;
    }
    float fx = clamp((u + 1.0) * 0.5 * (float)cW - 0.5, 0.0, (float)cW - 1.0);
    float fy = clamp((0.5 - v * 0.5) * (float)cH - 0.5, 0.0, (float)cH - 1.0);
    int x0 = (int)fx, y0 = (int)fy;
    int x1 = min(x0 + 1, (int)cW - 1), y1 = min(y0 + 1, (int)cH - 1);
    float tx = fx - (float)x0, ty = fy - (float)y0;
    float w00 = (1 - tx) * (1 - ty), w01 = tx * (1 - ty),
          w10 = (1 - tx) * ty,       w11 = tx * ty;

    float3 s = EquirectSampleRight(x0, x1, y0, y1, w00, w01, w10, w11);
    uint3 c = (uint3)clamp(s, 0.0, 255.0);
    gEqOut[dstIndex] = c.x | (c.y << 8) | (c.z << 16) | 0xFF000000u;
}
""";
}
