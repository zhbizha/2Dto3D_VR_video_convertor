namespace VRify.Core;

/// <summary>
/// Reprojects a flat (perspective) eye frame into a VR180 equirectangular view.
/// The flat frame is treated as an image on a virtual screen placed in front of
/// the viewer, spanning <see cref="ScreenFovDeg"/> horizontal degrees; area outside
/// the screen (and behind the viewer) is black.
///
/// Per eye: output is square (180° lon × 180° lat, equal degrees-per-pixel).
/// Full SBS frame = left eye equirect (x offset 0) | right eye equirect (x offset = eyeW).
/// </summary>
public sealed class EquirectProjector
{
    public double ScreenFovDeg { get; }

    // Precomputed per-column lon and per-row lat helpers (depend only on output size + fov)
    private int _lastW, _lastH;
    private double _lastFov;
    private float[] _tanLon = Array.Empty<float>();
    private float[] _invCosLon = Array.Empty<float>();
    private float[] _tanLat = Array.Empty<float>();

    public EquirectProjector(double screenFovDeg = 180.0)
    {
        // 180 = the flat 2D frame IS the full VR180 equirect half-sphere (no black
        // borders, no zoom-in). Lower values zoom in. The cap is 175 because at 180
        // tan(90) blows up and the perspective parameterisation breaks; 175 is close
        // enough that the flat frame fills 175/180 = 97% of the equirect.
        ScreenFovDeg = Math.Clamp(screenFovDeg, 60.0, 175.0);
    }

    /// <summary>
    /// Projects one flat eye frame into a square equirect eye at the given X offset
    /// inside a larger SBS destination frame.
    ///   src[srcXOffset .. srcXOffset+srcW) — eye columns inside src rows of stride srcStride
    ///   dst[dstXOffset .. dstXOffset+eyeW) — eye columns inside dst rows of stride dstFrameWidth
    /// </summary>
    public unsafe void Project(byte[] src, int srcXOffset, int srcW, int srcH, int srcStride,
                               byte[] dst, int dstXOffset, int dstFrameWidth,
                               int eyeW, int eyeH)
    {
        EnsureTables(eyeW, eyeH);

        double halfFov = ScreenFovDeg * Math.PI / 360.0;   // half horizontal FOV, rad
        float tanHalf = (float)Math.Tan(halfFov);
        // Vertical half-FOV follows source aspect on the tangent plane
        float ky = 1f / (tanHalf * srcH / srcW);
        float kx = 1f / tanHalf;

        int dstFrameStride = dstFrameWidth * 3;
        int dstXBase = dstXOffset * 3;
        int srcXBase = srcXOffset * 3;

        fixed (byte* pSrc = src, pDst = dst)
        fixed (float* pTanLon = _tanLon, pInvCos = _invCosLon, pTanLat = _tanLat)
        {
            for (int y = 0; y < eyeH; y++)
            {
                int dstRow = y * dstFrameStride + dstXBase;
                float tLat = pTanLat[y];
                for (int x = 0; x < eyeW; x++)
                {
                    // Ray dir = (sinLon·cosLat, sinLat, cosLon·cosLat); screen at z = 1
                    // Screen coords: u = tanLon / tanHalf,  v = (tanLat / cosLon) / tanHalfV
                    float u = pTanLon[x] * kx;
                    float v = tLat * pInvCos[x] * ky;

                    int o = dstRow + x * 3;
                    if (u <= -1f || u >= 1f || v <= -1f || v >= 1f)
                    {
                        pDst[o] = 0; pDst[o + 1] = 0; pDst[o + 2] = 0;
                        continue;
                    }

                    float fx = (u + 1f) * 0.5f * srcW - 0.5f;
                    float fy = (0.5f - v * 0.5f) * srcH - 0.5f;
                    if (fx < 0) fx = 0; else if (fx > srcW - 1) fx = srcW - 1;
                    if (fy < 0) fy = 0; else if (fy > srcH - 1) fy = srcH - 1;

                    int x0 = (int)fx, y0 = (int)fy;
                    int x1 = x0 + 1 < srcW ? x0 + 1 : x0;
                    int y1 = y0 + 1 < srcH ? y0 + 1 : y0;
                    float tx = fx - x0, ty = fy - y0;

                    int i00 = y0 * srcStride + srcXBase + x0 * 3;
                    int i01 = y0 * srcStride + srcXBase + x1 * 3;
                    int i10 = y1 * srcStride + srcXBase + x0 * 3;
                    int i11 = y1 * srcStride + srcXBase + x1 * 3;
                    float w00 = (1 - tx) * (1 - ty), w01 = tx * (1 - ty),
                          w10 = (1 - tx) * ty,       w11 = tx * ty;

                    pDst[o]     = (byte)Math.Clamp(pSrc[i00]     * w00 + pSrc[i01]     * w01 + pSrc[i10]     * w10 + pSrc[i11]     * w11, 0f, 255f);
                    pDst[o + 1] = (byte)Math.Clamp(pSrc[i00 + 1] * w00 + pSrc[i01 + 1] * w01 + pSrc[i10 + 1] * w10 + pSrc[i11 + 1] * w11, 0f, 255f);
                    pDst[o + 2] = (byte)Math.Clamp(pSrc[i00 + 2] * w00 + pSrc[i01 + 2] * w01 + pSrc[i10 + 2] * w10 + pSrc[i11 + 2] * w11, 0f, 255f);
                }
            }
        }
    }

    private void EnsureTables(int w, int h)
    {
        if (w == _lastW && h == _lastH && Math.Abs(_lastFov - ScreenFovDeg) < 1e-9) return;

        _tanLon = new float[w];
        _invCosLon = new float[w];
        for (int x = 0; x < w; x++)
        {
            double lon = ((x + 0.5) / w - 0.5) * Math.PI;   // -π/2 .. π/2
            double c = Math.Cos(lon);
            _tanLon[x] = (float)Math.Tan(lon);
            // cos→0 at edges: cap to avoid Infinity; v will exceed ±1 → black anyway
            _invCosLon[x] = (float)(1.0 / Math.Max(c, 1e-4));
        }
        _tanLat = new float[h];
        for (int y = 0; y < h; y++)
        {
            double lat = (0.5 - (y + 0.5) / h) * Math.PI;   // +π/2 (top) .. -π/2 (bottom)
            _tanLat[y] = (float)Math.Tan(lat);
        }
        _lastW = w; _lastH = h; _lastFov = ScreenFovDeg;
    }
}
