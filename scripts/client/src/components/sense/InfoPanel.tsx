// Live feature readout — ported from sense-human/src/components/InfoPanel.tsx.

import type { HumanAnalysis } from '../../vision/types'

function Meter({ label, value }: { label: string; value: number }) {
  return (
    <div className="meter">
      <span className="meter-label">{label}</span>
      <div className="meter-track">
        <div className="meter-fill" style={{ width: `${Math.min(100, value * 100)}%` }} />
      </div>
      <span className="meter-value">{(value * 100).toFixed(0)}</span>
    </div>
  )
}

function Row({ label, value, highlight }: { label: string; value: string; highlight?: boolean }) {
  return (
    <div className="srow">
      <span className="srow-label">{label}</span>
      <span className={highlight ? 'srow-value highlight' : 'srow-value'}>{value}</span>
    </div>
  )
}

export function InfoPanel({ analysis }: { analysis: HumanAnalysis | null }) {
  if (!analysis) return null
  const { face, body, fps, inferenceMs, present } = analysis

  return (
    <div className="sense-info">
      <div className="sense-section">
        <h3>Detection</h3>
        <Row label="Human" value={present ? 'detected' : 'not detected'} highlight={present} />
        <Row label="Analyzer FPS" value={fps.toFixed(0)} />
        <Row label="Inference" value={`${inferenceMs.toFixed(1)} ms`} />
      </div>

      {face && (
        <div className="sense-section">
          <h3>Face</h3>
          <Row label="Expression" value={face.expression} highlight />
          <Row label="Attention" value={face.attention} />
          <Row label="Blinking" value={face.blinking ? 'yes' : 'no'} />
          <Row
            label="Head yaw"
            value={`${face.headPose.yaw.toFixed(0)}° ${face.headPose.yaw > 8 ? '◀' : face.headPose.yaw < -8 ? '▶' : '•'}`}
          />
          <Row
            label="Head pitch"
            value={`${face.headPose.pitch.toFixed(0)}° ${face.headPose.pitch > 8 ? '▲' : face.headPose.pitch < -8 ? '▼' : '•'}`}
          />
          <Row label="Head roll" value={`${face.headPose.roll.toFixed(0)}°`} />
          <Meter label="Smile" value={face.smile} />
          <Meter label="Mouth open" value={face.jawOpen} />
          <Meter label="Brow raise" value={face.browRaise} />
        </div>
      )}

      {face && face.topBlendshapes.length > 0 && (
        <div className="sense-section">
          <h3>Top blendshapes</h3>
          {face.topBlendshapes.map((b) => (
            <Meter key={b.name} label={b.name} value={b.score} />
          ))}
        </div>
      )}

      {body && (
        <div className="sense-section">
          <h3>Body &amp; behavior</h3>
          <Row label="Activity" value={body.activity} highlight />
          <Row label="Posture" value={body.posture} />
          <Row label="Leaning" value={body.leaning} />
          <Row label="Shoulder tilt" value={`${body.shoulderTilt.toFixed(0)}°`} />
          <Row
            label="Hands raised"
            value={
              body.leftHandRaised && body.rightHandRaised ? 'both'
              : body.leftHandRaised ? 'left'
              : body.rightHandRaised ? 'right'
              : 'none'
            }
          />
          <Meter label="Movement" value={Math.min(1, body.movementEnergy * 40)} />
        </div>
      )}
    </div>
  )
}
