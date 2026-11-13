// "One CT scan -> every structure in 3D": NVIDIA NV-Segment-CT (VISTA3D) segments 117 structure types
// on one RTX 5060 Ti; a scanner sweep turns the CT slices into a 3D anatomy, layers peel away, then the
// honest check against expert-reviewed masks on CT scans the model never saw. Renders 4:5 and 9:16.
import React, {useEffect, useMemo, useState} from 'react';
import {AbsoluteFill, continueRender, delayRender, Easing, interpolate, staticFile, useCurrentFrame, useVideoConfig} from 'remotion';
import {ThreeCanvas} from '@remotion/three';
import {useThree} from '@react-three/fiber';
import * as THREE from 'three';
import {GLTFLoader} from 'three/examples/jsm/loaders/GLTFLoader.js';
import {loadFont as loadOnest} from '@remotion/google-fonts/Onest';
import {loadFont as loadMono} from '@remotion/google-fonts/JetBrainsMono';
import ANAT from '../public/anatomy.json';
import SLC from '../public/slices.json';
import DATA from '../public/data.json';

const {fontFamily: SANS} = loadOnest('normal', {weights: ['400', '600', '800'], subsets: ['latin']});
const {fontFamily: MONO} = loadMono('normal', {weights: ['400', '500'], subsets: ['latin']});

export const FPS = 30;
const TL = {hook: 3.4, stack: 5.6, sweepEnd: 10.0, peel: 14.0, results: 24.5, outro: 30.5, end: 36};
export const TOTAL = Math.round(TL.end * FPS);
const BG = '#05070C', TEXT = '#F4F1EA', DIM = '#8A929C', NV = '#76B900';

type Struct = {name: string; id: number; system: string; color: number[]; centroid: number[]; volume_ml: number; tris: number};
type Slice = {y: number; x: number[]; z: number[]; labels: number};
const STRUCT = (ANAT as unknown as {structures: Struct[]}).structures;
const EXT = (ANAT as unknown as {extent_cm: number[]}).extent_cm;
const SL = (SLC as unknown as {slices: Slice[]}).slices;
const D = DATA as unknown as {hero: {seconds: number; found: number; ct_slices: number};
  eval: {cases: number; median_dice: number; structure_checks: number; pct_above_090: number; min_seconds: number; max_seconds: number; median_seconds: number};
  repo: string};

const CLAMP = {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'} as const;
const OUT = Easing.bezier(0.16, 1, 0.3, 1);
const INOUT = Easing.bezier(0.65, 0, 0.35, 1);
const ramp = (s: number, a: number, b: number, e = OUT) => interpolate(s, [a, b], [0, 1], {...CLAMP, easing: e});
const TOP = EXT[1] / 2, BOT = -EXT[1] / 2;

// the scanner plane: top of the body at stack end -> bottom at sweepEnd
const sweepY = (s: number) => TOP + 1 - (TOP - BOT + 2) * ramp(s, TL.stack, TL.sweepEnd, INOUT);

// layer peel: which systems are visible when (opacity multiplier)
const SYSTEMS: [string, string][] = [['bone', 'SKELETON'], ['organ', 'ORGANS'], ['artery', 'ARTERIES'], ['vein', 'VEINS'],
  ['lung', 'LUNGS'], ['heart', 'HEART'], ['muscle', 'MUSCLES']];
const COUNT: Record<string, number> = {};
STRUCT.forEach((st) => { COUNT[st.system] = (COUNT[st.system] ?? 0) + 1; });
function layer(sys: string, s: number) {
  const P = TL.peel;
  const down = (a: number, lo = 0) => 1 - (1 - lo) * ramp(s, a, a + 1.0, INOUT);
  const up = (a: number) => ramp(s, a, a + 1.2, INOUT);
  let o = 1;
  if (sys === 'muscle') o = down(P + 1.0) + up(P + 8.0) * (1 - down(P + 1.0));
  if (sys === 'organ' || sys === 'lung') o = down(P + 3.0, 0.0) + up(P + 8.0) * (1 - down(P + 3.0, 0.0));
  if (sys === 'bone') o = down(P + 5.2, 0.1) + up(P + 8.0) * (1 - down(P + 5.2, 0.1));
  return o;
}

function useAssets() {
  const [geoms, setGeoms] = useState<Record<string, THREE.BufferGeometry> | null>(null);
  const [tex, setTex] = useState<{ct: THREE.Texture[]; mask: THREE.Texture[]} | null>(null);
  const [h1] = useState(() => delayRender('glb'));
  const [h2] = useState(() => delayRender('slices'));
  useEffect(() => {
    new GLTFLoader().load(staticFile('anatomy.glb'), (g) => {
      const m: Record<string, THREE.BufferGeometry> = {};
      g.scene.traverse((o) => {
        const mesh = o as THREE.Mesh;
        if (mesh.isMesh) { mesh.geometry.computeBoundingBox(); m[mesh.name] = mesh.geometry; }
      });
      setGeoms(m); continueRender(h1);
    });
    const L = new THREE.TextureLoader();
    const load = (p: string) => new Promise<THREE.Texture>((res) => L.load(staticFile(p), (t) => { t.colorSpace = THREE.SRGBColorSpace; res(t); }));
    const pad = (i: number) => String(i).padStart(2, '0');
    Promise.all([Promise.all(SL.map((_, i) => load(`slices/ct_${pad(i)}.jpg`))), Promise.all(SL.map((_, i) => load(`slices/mask_${pad(i)}.png`)))])
      .then(([ct, mask]) => { setTex({ct, mask}); continueRender(h2); });
  }, [h1, h2]);
  return {geoms, tex};
}
const sanitize = (n: string) => n.replace(/\s/g, '_').replace(/[[\]\.:\/]/g, '');

// camera: under the current slice looking up (radiological view) -> 3/4 front view -> slow orbit
const Rig: React.FC<{s: number; aspect: number; ySlice: number}> = ({s, aspect, ySlice}) => {
  const {camera, gl} = useThree();
  gl.localClippingEnabled = true;
  const cam = camera as THREE.PerspectiveCamera;
  const vfov = (cam.fov * Math.PI) / 180;
  const fitBody = Math.max((EXT[1] / 2) / Math.tan(vfov / 2), (EXT[0] / 2) / (Math.tan(vfov / 2) * aspect)) * (aspect < 0.7 ? 1.12 : 1.5);
  const fitSlice = (24 / (Math.tan(vfov / 2) * Math.min(1, aspect))) * 1.05;
  const t = ramp(s, TL.hook, TL.stack + 0.6, INOUT);
  const az = THREE.MathUtils.degToRad(28 + 360 * ramp(s, TL.stack + 0.6, TL.results + 3, (x) => x) + 40 * ramp(s, TL.results + 3, TL.end, (x) => x));
  const el = THREE.MathUtils.degToRad(9);
  const front = new THREE.Vector3(Math.sin(az) * Math.cos(el), Math.sin(el), Math.cos(az) * Math.cos(el));
  const dir = new THREE.Vector3(0, -1, 0).lerp(front, t).normalize();
  const dist = THREE.MathUtils.lerp(fitSlice, fitBody, t);
  // while the title sits on top, aim a bit higher so the body sits lower in the frame
  const lift = EXT[1] * (aspect < 0.7 ? 0.04 : 0.07) * (1 - ramp(s, TL.peel - 1, TL.peel + 0.5, INOUT));
  const target = new THREE.Vector3(0, THREE.MathUtils.lerp(ySlice, lift, t), 0);
  cam.up.copy(new THREE.Vector3(0, 0, 1).lerp(new THREE.Vector3(0, 1, 0), t).normalize());
  cam.position.copy(target.clone().add(dir.multiplyScalar(dist)));
  cam.lookAt(target);
  cam.updateProjectionMatrix();
  return null;
};

const Scene: React.FC<{geoms: Record<string, THREE.BufferGeometry>; tex: {ct: THREE.Texture[]; mask: THREE.Texture[]}; aspect: number}> = ({geoms, tex, aspect}) => {
  const f = useCurrentFrame();
  const s = f / FPS;
  const n = SL.length;
  const hookIdx = Math.min(n - 1, Math.floor(ramp(s, 0.2, TL.hook - 0.2, (x) => x) * (n - 1)));
  const ys = sweepY(s);
  const clip = useMemo(() => new THREE.Plane(new THREE.Vector3(0, 1, 0), 0), []);
  clip.constant = -ys;                                              // keep y >= ys (revealed above the scanner)
  const stack = ramp(s, TL.hook + 1.0, TL.stack + 0.3, OUT);        // ghost stack fades in once the camera is side-on
  const slicesGone = 1 - ramp(s, TL.sweepEnd - 0.4, TL.sweepEnd + 1.0);
  const sweepOn = s > TL.stack - 0.2 && s < TL.sweepEnd + 0.3;
  const nearest = SL.reduce((b, sl, i) => (Math.abs(sl.y - ys) < Math.abs(SL[b].y - ys) ? i : b), 0);
  return (
    <>
      <color attach="background" args={[BG]} />
      <Rig s={s} aspect={aspect} ySlice={SL[hookIdx].y} />
      <ambientLight intensity={0.55} />
      <directionalLight position={[30, 60, 80]} intensity={1.7} />
      <directionalLight position={[-40, 20, -60]} intensity={1.1} color="#7fb2ff" />
      <directionalLight position={[0, -50, 30]} intensity={0.35} />
      {/* CT slices: hook = only the current one, then a ghost stack that the scanner turns into anatomy */}
      {SL.map((sl, i) => {
        const inHook = s < TL.hook;
        let o = inHook ? (i === hookIdx ? 1 : 0) : (i === hookIdx ? 1 - 0.88 * stack : 0.12 * stack);
        if (!inHook && sl.y > ys) o *= 0.18;                       // above the scanner the slice gives way to 3D
        o *= slicesGone;
        if (o < 0.01) return null;
        const w = sl.x[1] - sl.x[0], d = sl.z[1] - sl.z[0];
        return (
          <mesh key={i} position={[(sl.x[0] + sl.x[1]) / 2, sl.y, (sl.z[0] + sl.z[1]) / 2]} rotation={[Math.PI / 2, 0, 0]} renderOrder={1}>
            <planeGeometry args={[w, d]} />
            <meshBasicMaterial map={tex.ct[i]} transparent opacity={o} side={THREE.DoubleSide} depthWrite={false}
              blending={inHook ? THREE.NormalBlending : THREE.AdditiveBlending} />
          </mesh>
        );
      })}
      {/* the scanner: coloured labels on the slice under the plane + a glowing ring */}
      {sweepOn && (
        <group position={[0, ys, 0]}>
          <mesh position={[(SL[nearest].x[0] + SL[nearest].x[1]) / 2, 0.05, (SL[nearest].z[0] + SL[nearest].z[1]) / 2]} rotation={[Math.PI / 2, 0, 0]} renderOrder={3}>
            <planeGeometry args={[SL[nearest].x[1] - SL[nearest].x[0], SL[nearest].z[1] - SL[nearest].z[0]]} />
            <meshBasicMaterial map={tex.mask[nearest]} transparent opacity={0.95} side={THREE.DoubleSide} depthWrite={false} />
          </mesh>
          <mesh rotation={[Math.PI / 2, 0, 0]} renderOrder={4}>
            <ringGeometry args={[25.5, 26.3, 128]} />
            <meshBasicMaterial color={NV} transparent opacity={0.9} side={THREE.DoubleSide} depthWrite={false} blending={THREE.AdditiveBlending} />
          </mesh>
          <mesh rotation={[Math.PI / 2, 0, 0]} renderOrder={2}>
            <circleGeometry args={[25.5, 128]} />
            <meshBasicMaterial color={NV} transparent opacity={0.06} side={THREE.DoubleSide} depthWrite={false} blending={THREE.AdditiveBlending} />
          </mesh>
        </group>
      )}
      {/* the anatomy */}
      {s >= TL.stack - 0.2 && STRUCT.map((st) => {
        const g = geoms[sanitize(st.name)] ?? geoms[st.name];
        if (!g) return null;
        const base = st.system === 'lung' ? 0.28 : st.system === 'muscle' ? 0.55 : 1;
        const o = base * layer(st.system, s);
        if (o < 0.01) return null;
        const c = new THREE.Color(st.color[0], st.color[1], st.color[2]);
        const glass = o < 0.999;
        return (
          // key changes with `glass`: three.js bakes `transparent` into the shader (OPAQUE define) and does not
          // recompile when it flips, so a fresh material is mounted instead
          <mesh key={`${st.id}${glass ? 'g' : 'o'}`} geometry={g} renderOrder={glass ? 5 : 0}>
            <meshStandardMaterial color={c} emissive={c} emissiveIntensity={0.18} roughness={0.42} metalness={0.08}
              transparent={glass} opacity={o} depthWrite={!glass} clippingPlanes={s < TL.sweepEnd + 0.2 ? [clip] : []} side={THREE.FrontSide} />
          </mesh>
        );
      })}
    </>
  );
};

const Chip: React.FC<{label: string; on: number}> = ({label, on}) => (
  <span style={{padding: '8px 16px', borderRadius: 30, border: `2px solid ${on > 0.5 ? NV : 'rgba(138,146,156,0.35)'}`,
    color: on > 0.5 ? TEXT : DIM, background: 'rgba(5,7,12,0.7)', opacity: 0.45 + 0.55 * on, whiteSpace: 'nowrap'}}>{label}</span>
);

const Overlay: React.FC<{W: number; H: number}> = ({W, H}) => {
  const f = useCurrentFrame();
  const s = f / FPS;
  const tall = H / W > 1.5;
  const pad = 56;
  const title = 90;
  // revealed structures counter and the measured time, while the scanner runs
  const ys = sweepY(s);
  const revealed = STRUCT.filter((st) => st.centroid[1] >= ys).length;
  const sweepP = ramp(s, TL.stack, TL.sweepEnd, INOUT);
  const hookO = 1 - ramp(s, TL.peel - 0.8, TL.peel);
  const resO = ramp(s, TL.results, TL.results + 0.8) * (1 - ramp(s, TL.outro - 0.6, TL.outro));
  const outO = ramp(s, TL.outro, TL.outro + 0.8);
  const peelO = ramp(s, TL.peel, TL.peel + 0.6) * (1 - ramp(s, TL.results - 0.6, TL.results));
  const second = ramp(s, TL.hook - 0.2, TL.hook + 0.5);
  return (
    <AbsoluteFill style={{fontFamily: SANS, color: TEXT}}>
      {hookO > 0.01 && (
        <div style={{position: 'absolute', left: pad, right: pad, top: tall ? 120 : 64, opacity: hookO, fontWeight: 800, fontSize: title,
          lineHeight: 1.0, letterSpacing: -3, textShadow: '0 4px 30px rgba(0,0,0,0.9)'}}>
          One CT scan.
          <div style={{color: NV, opacity: second, transform: `translateY(${(1 - second) * 20}px)`}}>Every structure in 3D.</div>
          <div style={{fontFamily: MONO, fontWeight: 400, fontSize: 26, letterSpacing: 0, marginTop: 20, color: DIM}}>
            {D.hero.ct_slices} slices · 1.5 mm · neck to thighs
          </div>
        </div>
      )}
      {s > TL.stack - 0.3 && s < TL.peel && (
        <div style={{position: 'absolute', left: pad, right: pad, bottom: tall ? 170 : 70, display: 'flex', justifyContent: 'space-between',
          fontFamily: MONO, fontSize: 28, opacity: ramp(s, TL.stack - 0.3, TL.stack + 0.3) * (1 - ramp(s, TL.peel - 0.6, TL.peel))}}>
          <span style={{padding: '10px 18px', borderRadius: 30, background: 'rgba(5,7,12,0.78)', border: `2px solid ${NV}`}}>
            ● NV-Segment-CT · {(D.hero.seconds * sweepP).toFixed(1)} s
          </span>
          <span style={{padding: '10px 18px', borderRadius: 30, background: 'rgba(5,7,12,0.78)'}}>{Math.min(revealed, D.hero.found)} structures</span>
        </div>
      )}
      {peelO > 0.01 && (
        <div style={{position: 'absolute', left: pad, right: pad, bottom: tall ? 170 : 70, display: 'flex', flexWrap: 'wrap', gap: 12,
          fontFamily: MONO, fontSize: 24, opacity: peelO}}>
          {SYSTEMS.filter(([k]) => COUNT[k]).map(([k, label]) => <Chip key={k} label={`${label} ${COUNT[k]}`} on={layer(k, s)} />)}
        </div>
      )}
      {resO > 0.01 && (
        <AbsoluteFill style={{opacity: resO, background: `linear-gradient(180deg, rgba(5,7,12,0.92) 0%, rgba(5,7,12,0.55) ${tall ? 45 : 55}%, rgba(5,7,12,0.0) 75%)`}}>
          <div style={{position: 'absolute', left: pad, right: pad, top: tall ? 130 : 70}}>
            <div style={{fontFamily: MONO, fontSize: 28, color: DIM}}>vs expert-reviewed masks · {D.eval.cases} CT scans it never saw</div>
            <div style={{fontWeight: 800, fontSize: tall ? 250 : 220, letterSpacing: -10, lineHeight: 1, marginTop: 10}}>
              {D.eval.median_dice.toFixed(2)}
            </div>
            <div style={{fontWeight: 600, fontSize: 44, marginTop: 4}}>median Dice <span style={{color: DIM, fontWeight: 400}}>· 1.00 = perfect overlap</span></div>
            <div style={{fontFamily: MONO, fontSize: 28, lineHeight: 1.6, marginTop: 28}}>
              {D.eval.structure_checks.toLocaleString('en-US')} structure checks · {D.eval.pct_above_090}% above 0.90<br />
              <span style={{color: NV}}>{D.eval.min_seconds.toFixed(0)}–{D.eval.max_seconds.toFixed(0)} s per scan</span> · one RTX 5060 Ti · TensorRT
            </div>
          </div>
        </AbsoluteFill>
      )}
      {outO > 0.01 && (
        <AbsoluteFill style={{opacity: outO, background: `rgba(5,7,12,${0.9 * outO})`}}>
          <div style={{position: 'absolute', left: pad, right: pad, top: tall ? 380 : 230, fontWeight: 800, fontSize: tall ? 100 : 92, letterSpacing: -3, lineHeight: 1.02}}>
            One CT.<div style={{color: NV}}>{D.hero.found} structures.</div>Seconds.
          </div>
          <div style={{position: 'absolute', left: pad, right: pad, top: tall ? 820 : 620, fontFamily: MONO, fontSize: 30, lineHeight: 1.7}}>
            {D.repo}
          </div>
          <div style={{position: 'absolute', left: pad, right: pad, bottom: tall ? 200 : 80, fontFamily: MONO, fontSize: 23, lineHeight: 1.65, color: DIM}}>
            NVIDIA NV-Segment-CT (VISTA3D) · MONAI · TensorRT<br />
            Data: TotalSegmentator v3 (CC BY 4.0), scans published after the model<br />
            Research demo · not a medical device<br />Zalkarbek Arapbaev · physician &amp; CV engineer
          </div>
        </AbsoluteFill>
      )}
    </AbsoluteFill>
  );
};

export const CT: React.FC = () => {
  const {width: W, height: H} = useVideoConfig();
  const {geoms, tex} = useAssets();
  return (
    <AbsoluteFill style={{background: BG}}>
      {geoms && tex && (
        <ThreeCanvas width={W} height={H} camera={{fov: 30, near: 1, far: 2000}} gl={{antialias: true, preserveDrawingBuffer: true}}>
          <Scene geoms={geoms} tex={tex} aspect={W / H} />
        </ThreeCanvas>
      )}
      <Overlay W={W} H={H} />
    </AbsoluteFill>
  );
};
