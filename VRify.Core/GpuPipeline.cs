using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Vortice.D3DCompiler;
using Vortice.Direct3D;
using Vortice.Direct3D11;
using Vortice.DXGI;

namespace VRify.Core;

/// <summary>
/// D3D11 compute pipeline: runs the whole post-depth stereo synthesis on the GPU —
/// temporal smooth → guided refine → median → forward-warp → thin-fill → pyramid
/// inpaint → SBS pack / equirect project → readback. Replaces the CPU path in
/// <see cref="VrConverter"/> when a device is available; null (CPU fallback) otherwise.
/// Not thread-safe: drive from a single thread (the warp stage).
/// </summary>
public sealed class GpuPipeline : IDisposable
{
    // ---- constant buffer layout (must match GpuShaders.CB: 16 x 4B = 64B) ----
    [StructLayout(LayoutKind.Sequential)]
    private struct CBData
    {
        public uint W, H;
        public float MaxShift, ZTol, AStat, AMot, Eps;
        public uint Radius, First, EyeW, EyeH;
        public float Kx, Ky;
        public uint Pad0, Pad1, Pad2;
    }

    private sealed class Slot
    {
        public ID3D11Buffer Buffer = null!;
        public ID3D11ShaderResourceView Srv = null!;
        public ID3D11UnorderedAccessView Uav = null!;
        public void Dispose()
        {
            Uav?.Dispose(); Srv?.Dispose(); Buffer?.Dispose();
            Uav = null!; Srv = null!; Buffer = null!;
        }
    }

    private readonly ID3D11Device _dev;
    private readonly ID3D11DeviceContext _ctx;

    private readonly ID3D11ComputeShader _smooth, _guideBuild, _boxH, _boxV, _abPass,
        _guidedFinal, _median3, _collapse, _hardStep, _warpGather, _thinFill, _bgFill, _pyInit, _pyDown, _pyUp,
        _pyFinalize, _packSbs, _equirectLeft, _equirectRight;

    private readonly ID3D11Buffer _cb;
    private readonly ID3D11ShaderResourceView[] _nullSrv = new ID3D11ShaderResourceView[5];
    private readonly ID3D11UnorderedAccessView[] _nullUav = new ID3D11UnorderedAccessView[2];

    public string Info { get; }

    // ---- source-resolution resources ----
    private int _w, _h, _levels, _eyeSize;
    private bool _vr;
    private readonly Slot[] _rgb = { new(), new() };   // dynamic RGB24 uplink (ping-pong: prev frame)
    private readonly Slot _depth = new();              // dynamic structured float
    private readonly Slot[] _hist = { new(), new() };  // depth history ping-pong (smooth in → next frame)
    private readonly Slot _ref = new();                // median output (refined depth)
    private readonly Slot _gA = new(), _gTmp = new(), _gMeans = new(), _gQ = new();
    private readonly Slot _packA = new(), _packB = new();
    private readonly Slot _warpDisp = new();           // winner disparity per destination px (-1 = hole)
    private Slot[] _pyA = Array.Empty<Slot>(), _pyB = Array.Empty<Slot>();

    // ---- output / readback ----
    private int _outW, _outH, _outBytes;
    private readonly Slot _out = new();
    private ID3D11Buffer _staging = null!;

    // ---- equirect tables (VR180) ----
    private readonly Slot _tabLon = new(), _tabCos = new(), _tabLat = new();
    private int _tabEye = -1;
    private double _tabFov = -1;
    private float _kx, _ky;

    // ---- per-conversion state ----
    private float _maxShift, _zTol, _aStat, _aMot;
    private bool _first = true;
    private int _par;

    /// <summary>
    /// Use the background-mirror disocclusion fill (default) instead of the legacy
    /// thin-row fill + push-pull blur. Mirrors <see cref="StereoSynthesizer.BackgroundFill"/>.
    /// </summary>
    public bool BackgroundFill { get; set; } = true;

    /// <summary>Creates the device + shaders. On any failure returns null and reports why in <paramref name="info"/>.</summary>
    public static GpuPipeline? TryCreate(int gpuDeviceId, out string info)
    {
        if (Environment.GetEnvironmentVariable("VRIFY_NO_GPU") is { Length: > 0 })
        {
            info = "CPU (forced by VRIFY_NO_GPU)";
            return null;
        }
        try
        {
            var p = new GpuPipeline(gpuDeviceId);
            info = p.Info;
            return p;
        }
        catch (Exception ex)
        {
            info = "CPU (GPU init failed: " + ex.Message + ")";
            return null;
        }
    }

    private GpuPipeline(int gpuDeviceId)
    {
        // ---- pick a DXGI adapter: NVIDIA preferred, else by requested index ----
        using var factory = DXGI.CreateDXGIFactory1<IDXGIFactory1>();
        var adapters = new List<IDXGIAdapter1>();
        for (uint i = 0; ; i++)
        {
            if (factory.EnumAdapters1(i, out var a).Failure) break;
            adapters.Add(a);
        }
        if (adapters.Count == 0) throw new InvalidOperationException("no DXGI adapters");

        var nvidia = adapters.Where(a => a.Description.VendorId == 0x10DE).ToList();
        var pool = nvidia.Count > 0 ? nvidia : adapters;
        var sel = pool[Math.Clamp(gpuDeviceId, 0, pool.Count - 1)];

        var ad = sel.Description;
        string name = ad.Description?.Trim() ?? "unknown";
        ulong vramMb = (ulong)ad.DedicatedVideoMemory / (1024UL * 1024UL);

        var levels = new[] { FeatureLevel.Level_11_0 };
        ID3D11Device dev = null!;
        ID3D11DeviceContext ctx = null!;
        var hr = D3D11.D3D11CreateDevice(sel, DriverType.Hardware, DeviceCreationFlags.None,
            levels, out dev, out ctx);
        if (hr.Failure)
            hr = D3D11.D3D11CreateDevice(IntPtr.Zero, DriverType.Hardware, DeviceCreationFlags.None,
                levels, out dev, out ctx);
        foreach (var a in adapters) a.Dispose();
        if (hr.Failure)
            throw new InvalidOperationException("D3D11CreateDevice failed: " + hr.Description);

        _dev = dev;
        _ctx = ctx;
        Info = $"D3D11: {name} ({vramMb} MB)";

        _smooth = Compile("Smooth");
        _guideBuild = Compile("GuideBuild");
        _boxH = Compile("BoxH");
        _boxV = Compile("BoxV");
        _abPass = Compile("ABPass");
        _guidedFinal = Compile("GuidedFinal");
        _median3 = Compile("Median3");
        _collapse = Compile("CollapseRamps");
        _hardStep = Compile("HardStepPass");
        _warpGather = Compile("WarpGather");
        _thinFill = Compile("ThinFill");
        _bgFill = Compile("BgFill");
        _pyInit = Compile("PyInit");
        _pyDown = Compile("PyDown");
        _pyUp = Compile("PyUp");
        _pyFinalize = Compile("PyFinalize");
        _packSbs = Compile("PackSbs");
        _equirectLeft = Compile("EquirectProjectLeft");
        _equirectRight = Compile("EquirectProjectRight");

        _cb = NewBuffer(64, ResourceUsage.Dynamic, BindFlags.ConstantBuffer,
            CpuAccessFlags.Write, ResourceOptionFlags.None, 0);
    }

    private ID3D11ComputeShader Compile(string entry)
    {
        try
        {
            var bytes = Compiler.Compile(GpuShaders.Source, entry, "vrify.hlsl", "cs_5_0",
                ShaderFlags.OptimizationLevel3);
            return _dev.CreateComputeShader(bytes.Span);
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException($"shader '{entry}' failed to compile: {ex.Message}", ex);
        }
    }

    // ==================================================================
    // Configuration (once per conversion, before the stage threads start)
    // ==================================================================
    public void Configure(int w, int h, bool vr, int eyeSize, double screenFovDeg,
        float strengthPercent, float temporalSmoothing)
    {
        ClearBindings();
        EnsureResources(w, h, vr, eyeSize);

        float strength = Math.Clamp(strengthPercent, 0f, 10f);
        _maxShift = strength / 100f * w;
        _zTol = Math.Max(0.5f, _maxShift * 0.03f);

        float s = Math.Clamp(temporalSmoothing, 0f, 0.95f);
        _aStat = 1f - s;
        _aMot = MathF.Min(0.9f, _aStat + 0.6f);

        if (vr) SetupEquirect(eyeSize, screenFovDeg, w, h);
        ResetTemporal();
    }

    /// <summary>Clears temporal state — call at the start of every conversion.</summary>
    public void ResetTemporal()
    {
        _first = true;
        _par = 0;
    }

    // ==================================================================
    // Per-frame: rgb (w*h*3) + depth (w*h floats) -> sbsOut (outW*outH*4, RGBA)
    // ==================================================================
    public void Render(byte[] rgb, float[] depth, byte[] sbsOut)
    {
        int w = _w, h = _h;
        int gx = (w + 7) / 8, gy = (h + 7) / 8;
        int cur = _par, prv = 1 - _par;

        ClearBindings();
        UploadBytes(_rgb[cur].Buffer, rgb);
        UploadFloats(_depth.Buffer, depth);

        // 1) temporal smooth → _hist[cur]
        Dispatch(gx, gy, _smooth, new CBData
        {
            W = (uint)w, H = (uint)h, AStat = _aStat, AMot = _aMot, First = _first ? 1u : 0u,
        }, new[] { _depth.Srv, _hist[prv].Srv, _rgb[cur].Srv, _rgb[prv].Srv }, new[] { _hist[cur].Uav });

        // 2) edge re-snap + median 3x3 → _ref. Mirrors DepthEstimator.RefineEdges: the
        //    gated ramp collapse (default) or, for A/B, the legacy guided filter that
        //    measurably WIDENED silhouette transitions.
        if (DepthEstimator.SharpenEdges)
        {
            Dispatch((w + 63) / 64, h, _collapse, new CBData { W = (uint)w, H = (uint)h },
                new[] { _hist[cur].Srv }, new[] { _gQ.Uav });
            // Hard-step: turn the last few px of bilinear-upsample AA into a real step
            // (mirrors DepthEstimator.HardStepPass). In-place on _gQ (UAV barrier is
            // inserted between dispatches; SRV+UAV on the same buffer is fine when used
            // in separate Dispatch calls).
            Dispatch((w + 63) / 64, h, _hardStep, new CBData { W = (uint)w, H = (uint)h },
                new[] { _gQ.Srv }, new[] { _gQ.Uav });
        }
        else
        {
            Dispatch(gx, gy, _guideBuild, new CBData { W = (uint)w, H = (uint)h },
                new[] { _hist[cur].Srv, _rgb[cur].Srv }, new[] { _gA.Uav });
            Dispatch((w + 63) / 64, h, _boxH, new CBData { W = (uint)w, H = (uint)h, Radius = 14 },
                new[] { _gA.Srv }, new[] { _gTmp.Uav });
            Dispatch(w, (h + 63) / 64, _boxV, new CBData { W = (uint)w, H = (uint)h, Radius = 14 },
                new[] { _gTmp.Srv }, new[] { _gMeans.Uav });
            Dispatch(gx, gy, _abPass, new CBData { W = (uint)w, H = (uint)h, Eps = 0.004f },
                null, new[] { _gMeans.Uav });
            Dispatch((w + 63) / 64, h, _boxH, new CBData { W = (uint)w, H = (uint)h, Radius = 14 },
                new[] { _gMeans.Srv }, new[] { _gTmp.Uav });
            Dispatch(w, (h + 63) / 64, _boxV, new CBData { W = (uint)w, H = (uint)h, Radius = 14 },
                new[] { _gTmp.Srv }, new[] { _gMeans.Uav });
            Dispatch(gx, gy, _guidedFinal, new CBData { W = (uint)w, H = (uint)h },
                new[] { _gMeans.Srv, _gA.Srv }, new[] { _gQ.Uav });
        }
        Dispatch(gx, gy, _median3, new CBData { W = (uint)w, H = (uint)h },
            new[] { _gQ.Srv }, new[] { _ref.Uav });

        // History for the next frame: frame 0 keeps the raw depth (mirrors the CPU
        // smoother, whose first-frame snapshot happens before refinement); frames >= 1
        // blend against the refined depth written into _hist[cur].
        if (_first)
        {
            // _hist[cur] already holds the raw frame-0 depth written by Smooth.
        }
        else
        {
            ClearBindings();
            _ctx.CopyResource(_hist[cur].Buffer, _ref.Buffer);
        }

        // 3) forward warp + disocclusion fill
        Dispatch(gx, gy, _warpGather, new CBData
        {
            W = (uint)w, H = (uint)h, MaxShift = _maxShift, ZTol = _zTol,
        }, new[] { _ref.Srv, _rgb[cur].Srv }, new[] { _packA.Uav, _warpDisp.Uav });

        if (BackgroundFill)
            Dispatch(gx, gy, _bgFill, new CBData { W = (uint)w, H = (uint)h, MaxShift = _maxShift },
                new[] { _packA.Srv, _warpDisp.Srv, _rgb[cur].Srv }, new[] { _packB.Uav });
        else
            Dispatch(gx, gy, _thinFill, new CBData { W = (uint)w, H = (uint)h },
                new[] { _packA.Srv }, new[] { _packB.Uav });

        // 4) push-pull pyramid inpaint → final right eye in _packA
        Dispatch(gx, gy, _pyInit, new CBData { W = (uint)w, H = (uint)h },
            new[] { _packB.Srv }, new[] { _pyA[0].Uav });
        for (int l = 1; l < _levels; l++)
        {
            int cw = Math.Max(1, w >> l), ch = Math.Max(1, h >> l);
            int fw = Math.Max(1, w >> (l - 1)), fh = Math.Max(1, h >> (l - 1));
            Dispatch((cw + 7) / 8, (ch + 7) / 8, _pyDown,
                new CBData { W = (uint)cw, H = (uint)ch, Kx = fw, Ky = fh },
                new[] { _pyA[l - 1].Srv }, new[] { _pyA[l].Uav });
        }
        for (int j = _levels - 2; j >= 0; j--)
        {
            int fw = Math.Max(1, w >> j), fh = Math.Max(1, h >> j);
            int cw2 = Math.Max(1, w >> (j + 1)), ch2 = Math.Max(1, h >> (j + 1));
            var coarse = j == _levels - 2 ? _pyA[_levels - 1].Srv : _pyB[j + 1].Srv;
            Dispatch((fw + 7) / 8, (fh + 7) / 8, _pyUp,
                new CBData { W = (uint)fw, H = (uint)fh, Kx = cw2, Ky = ch2 },
                new[] { _pyA[j].Srv, coarse }, new[] { _pyB[j].Uav });
        }
        Dispatch(gx, gy, _pyFinalize, new CBData { W = (uint)w, H = (uint)h },
            new[] { _pyB[0].Srv, _packB.Srv }, new[] { _packA.Uav });

        // 5) pack SBS (RGBA) or VR180 equirect (two eye passes)
        if (!_vr)
        {
            Dispatch(gx, gy, _packSbs, new CBData { W = (uint)w, H = (uint)h, EyeW = (uint)(2 * w) },
                new[] { _rgb[cur].Srv, _packA.Srv }, new[] { _out.Uav });
        }
        else
        {
            int ex = (_eyeSize + 7) / 8;
            var cb = new CBData
            {
                W = (uint)w, H = (uint)h, EyeW = (uint)_eyeSize, EyeH = (uint)_eyeSize,
                Kx = _kx, Ky = _ky, Pad0 = 0, Pad1 = 0,
            };
            var srvs = new[] { _rgb[cur].Srv, _packA.Srv, _tabLon.Srv, _tabCos.Srv, _tabLat.Srv };
            // Two specialised entry points (left/right) hard-code the source SRV slot,
            // so the CB does not need an "eye select" field.
            Dispatch(ex, ex, _equirectLeft,  cb, srvs, new[] { _out.Uav });
            cb.Pad0 = (uint)_eyeSize;
            Dispatch(ex, ex, _equirectRight, cb, srvs, new[] { _out.Uav });
        }

        // 6) readback
        ClearBindings();
        _ctx.CopyResource(_staging, _out.Buffer);
        var m = _ctx.Map(_staging, MapMode.Read);
        unsafe { new ReadOnlySpan<byte>((void*)m.DataPointer, _outBytes).CopyTo(sbsOut); }
        _ctx.Unmap(_staging);

        // 7) advance state
        _par = prv;
        _first = false;
    }

    public void Dispose()
    {
        foreach (var cs in new[]
        {
            _smooth, _guideBuild, _boxH, _boxV, _abPass, _guidedFinal, _median3, _collapse,
            _warpGather, _thinFill, _bgFill, _pyInit, _pyDown, _pyUp, _pyFinalize, _packSbs,
            _equirectLeft, _equirectRight,
        })
            cs?.Dispose();

        _cb?.Dispose();
        foreach (var s in _rgb) s.Dispose();
        _depth.Dispose(); _ref.Dispose();
        foreach (var s in _hist) s.Dispose();
        _gA.Dispose(); _gTmp.Dispose(); _gMeans.Dispose(); _gQ.Dispose();
        _packA.Dispose(); _packB.Dispose();
        _warpDisp.Dispose();
        foreach (var s in _pyA) s.Dispose();
        foreach (var s in _pyB) s.Dispose();
        _out.Dispose();
        _staging?.Dispose();
        _tabLon.Dispose(); _tabCos.Dispose(); _tabLat.Dispose();
        _ctx?.Dispose();
        _dev?.Dispose();
    }

    // ==================================================================
    // Resource management
    // ==================================================================
    private bool _allocated;

    private void EnsureResources(int w, int h, bool vr, int eyeSize)
    {
        if (_allocated && w == _w && h == _h && vr == _vr && eyeSize == _eyeSize) return;
        if (_allocated) ReleaseResources();

        _w = w; _h = h; _vr = vr; _eyeSize = eyeSize;
        int n = w * h;

        // RGB24 upload buffers: 4-byte aligned, +4 slack (LoadRGB touches 1 word past the last pixel).
        int rgbBytes = ((w * h * 3 + 3) & ~3) + 4;
        for (int i = 0; i < 2; i++)
        {
            _rgb[i].Buffer = NewBuffer(rgbBytes, ResourceUsage.Dynamic, BindFlags.ShaderResource,
                CpuAccessFlags.Write, ResourceOptionFlags.None, 0);
            _rgb[i].Srv = NewUintSrv(_rgb[i].Buffer, rgbBytes / 4);
        }

        _depth.Buffer = NewBuffer(n * 4, ResourceUsage.Dynamic, BindFlags.ShaderResource,
            CpuAccessFlags.Write, ResourceOptionFlags.BufferStructured, 4);
        _depth.Srv = NewStructuredSrv(_depth.Buffer, n);

        for (int i = 0; i < 2; i++)
            NewStructuredSlot(_hist[i], n, 4, uav: true);
        NewStructuredSlot(_ref, n, 4, uav: true);
        NewStructuredSlot(_gA, n, 16, uav: true);
        NewStructuredSlot(_gTmp, n, 16, uav: true);
        NewStructuredSlot(_gMeans, n, 16, uav: true);
        NewStructuredSlot(_gQ, n, 4, uav: true);
        NewStructuredSlot(_packA, n, 4, uav: true);
        NewStructuredSlot(_packB, n, 4, uav: true);
        NewStructuredSlot(_warpDisp, n, 4, uav: true);

        int levels = 1;
        while ((w >> levels) > 4 && (h >> levels) > 4) levels++;
        levels = Math.Max(2, levels);
        _levels = levels;
        _pyA = new Slot[levels];
        _pyB = new Slot[levels - 1];
        for (int l = 0; l < levels; l++)
        {
            int cnt = Math.Max(1, w >> l) * Math.Max(1, h >> l);
            _pyA[l] = new Slot();
            NewStructuredSlot(_pyA[l], cnt, 16, uav: true);
            if (l < levels - 1)
            {
                _pyB[l] = new Slot();
                NewStructuredSlot(_pyB[l], cnt, 16, uav: true);
            }
        }

        _outW = vr ? eyeSize * 2 : w * 2;
        _outH = vr ? eyeSize : h;
        _outBytes = _outW * _outH * 4;
        NewStructuredSlot(_out, _outW * _outH, 4, uav: true);
        _out.Srv?.Dispose();
        _out.Srv = null!;
        _staging = NewBuffer(_outBytes, ResourceUsage.Staging, BindFlags.None,
            CpuAccessFlags.Read, ResourceOptionFlags.None, 0);

        _allocated = true;
    }

    private void ReleaseResources()
    {
        foreach (var s in _rgb) s.Dispose();
        _depth.Dispose(); _ref.Dispose();
        foreach (var s in _hist) s.Dispose();
        _gA.Dispose(); _gTmp.Dispose(); _gMeans.Dispose(); _gQ.Dispose();
        _packA.Dispose(); _packB.Dispose();
        _warpDisp.Dispose();
        foreach (var s in _pyA) s.Dispose();
        foreach (var s in _pyB) s.Dispose();
        _out.Dispose();
        _staging?.Dispose();
        _staging = null!;
        _allocated = false;
    }

    private void SetupEquirect(int eyeSize, double screenFovDeg, int srcW, int srcH)
    {
        if (eyeSize != _tabEye || Math.Abs(screenFovDeg - _tabFov) > 1e-9)
        {
            _tabLon.Dispose(); _tabCos.Dispose(); _tabLat.Dispose();

            var lon = new float[eyeSize];
            var cos = new float[eyeSize];
            for (int x = 0; x < eyeSize; x++)
            {
                double l = ((x + 0.5) / eyeSize - 0.5) * Math.PI;
                double c = Math.Cos(l);
                lon[x] = (float)Math.Tan(l);
                cos[x] = (float)(1.0 / Math.Max(c, 1e-4));
            }
            var lat = new float[eyeSize];
            for (int y = 0; y < eyeSize; y++)
                lat[y] = (float)Math.Tan((0.5 - (y + 0.5) / eyeSize) * Math.PI);

            NewFloatTable(_tabLon, lon);
            NewFloatTable(_tabCos, cos);
            NewFloatTable(_tabLat, lat);
            _tabEye = eyeSize;
            _tabFov = screenFovDeg;
        }

        double halfFov = Math.Clamp(screenFovDeg, 60.0, 178.0) * Math.PI / 360.0;
        float tanHalf = (float)Math.Tan(halfFov);
        _kx = 1f / tanHalf;
        _ky = 1f / (tanHalf * srcH / srcW);
    }

    // ==================================================================
    // Low-level helpers
    // ==================================================================
    private void Dispatch(int gx, int gy, ID3D11ComputeShader shader, CBData cb,
        ID3D11ShaderResourceView[]? srvs, ID3D11UnorderedAccessView[]? uavs)
    {
        // Unbind previous bindings first: a resource written as UAV in the previous
        // dispatch must not stay bound when it is read as SRV here (D3D11 hazard).
        ClearBindings();
        _ctx.CSSetShader(shader);
        WriteCb(cb);
        _ctx.CSSetConstantBuffer(0, _cb);
        if (srvs != null) _ctx.CSSetShaderResources(0, srvs);
        if (uavs != null) _ctx.CSSetUnorderedAccessViews(0, uavs);
        _ctx.Dispatch((uint)gx, (uint)gy, 1);
    }

    private void ClearBindings()
    {
        _ctx.CSSetShaderResources(0, _nullSrv);
        _ctx.CSSetUnorderedAccessViews(0, _nullUav);
    }

    private unsafe void WriteCb(in CBData cb)
    {
        var m = _ctx.Map(_cb, MapMode.WriteDiscard);
        Unsafe.WriteUnaligned((void*)m.DataPointer, cb);
        _ctx.Unmap(_cb);
    }

    private unsafe void UploadBytes(ID3D11Buffer buf, byte[] src)
    {
        var m = _ctx.Map(buf, MapMode.WriteDiscard);
        fixed (byte* p = src)
            Unsafe.CopyBlockUnaligned((void*)m.DataPointer, p, (uint)src.Length);
        _ctx.Unmap(buf);
    }

    private unsafe void UploadFloats(ID3D11Buffer buf, float[] src)
    {
        var m = _ctx.Map(buf, MapMode.WriteDiscard);
        fixed (float* p = src)
            Unsafe.CopyBlockUnaligned((void*)m.DataPointer, p, (uint)(src.Length * 4));
        _ctx.Unmap(buf);
    }

    private ID3D11Buffer NewBuffer(int size, ResourceUsage usage, BindFlags bind,
        CpuAccessFlags cpu, ResourceOptionFlags opt, uint stride)
        => _dev.CreateBuffer((uint)size, bind, usage, cpu, opt, stride);

    private void NewStructuredSlot(Slot slot, int count, int stride, bool uav)
    {
        var bind = BindFlags.ShaderResource | (uav ? BindFlags.UnorderedAccess : BindFlags.None);
        slot.Buffer = NewBuffer(count * stride, ResourceUsage.Default, bind,
            CpuAccessFlags.None, ResourceOptionFlags.BufferStructured, (uint)stride);
        slot.Srv = NewStructuredSrv(slot.Buffer, count);
        if (uav) slot.Uav = NewStructuredUav(slot.Buffer, count);
    }

    private void NewFloatTable(Slot slot, float[] data)
    {
        slot.Buffer = NewBuffer(data.Length * 4, ResourceUsage.Default, BindFlags.ShaderResource,
            CpuAccessFlags.None, ResourceOptionFlags.BufferStructured, 4);
        slot.Srv = NewStructuredSrv(slot.Buffer, data.Length);
        _ctx.UpdateSubresource(data, slot.Buffer);
    }

    private ID3D11ShaderResourceView NewStructuredSrv(ID3D11Buffer buf, int count)
    {
        var d = new ShaderResourceViewDescription
        {
            Format = Format.Unknown,
            ViewDimension = ShaderResourceViewDimension.Buffer,
        };
        d.Buffer.FirstElement = 0;
        d.Buffer.NumElements = (uint)count;
        return _dev.CreateShaderResourceView(buf, d);
    }

    private ID3D11ShaderResourceView NewUintSrv(ID3D11Buffer buf, int elements)
    {
        var d = new ShaderResourceViewDescription
        {
            Format = Format.R32_UInt,
            ViewDimension = ShaderResourceViewDimension.Buffer,
        };
        d.Buffer.FirstElement = 0;
        d.Buffer.NumElements = (uint)elements;
        return _dev.CreateShaderResourceView(buf, d);
    }

    private ID3D11UnorderedAccessView NewStructuredUav(ID3D11Buffer buf, int count)
    {
        var d = new UnorderedAccessViewDescription
        {
            Format = Format.Unknown,
            ViewDimension = UnorderedAccessViewDimension.Buffer,
        };
        d.Buffer.FirstElement = 0;
        d.Buffer.NumElements = (uint)count;
        return _dev.CreateUnorderedAccessView(buf, d);
    }
}
