import React from 'react';
import {Composition} from 'remotion';
import {CT, TOTAL, FPS} from './CT';

// one scene, two layouts: 4:5 for X/LinkedIn, 9:16 for YouTube Shorts
export const Root: React.FC = () => (
  <>
    <Composition id="CT45" component={CT} durationInFrames={TOTAL} fps={FPS} width={1080} height={1350} />
    <Composition id="CT916" component={CT} durationInFrames={TOTAL} fps={FPS} width={1080} height={1920} />
  </>
);
