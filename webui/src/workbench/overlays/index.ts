// SPDX-FileCopyrightText: 2026 Dpt. <https://www.dpt.co>
// SPDX-License-Identifier: GPL-3.0-or-later

// Spatial pipeline data over the splats (the design center): camera
// frustums (auto + manual, selectable), editable volume boxes with a
// translate/scale manipulator + always-on tags (heights are invisible
// in 2D; here they are geometry, a tag, and panel numerics), the focus
// and camera manipulators, the origin grid and the floor plane,
// exemplar-crop source
// markers. Each overlay is parented in the frame of the file it draws
// (alignment.ts names the two): the volume boxes, the focus manipulator,
// the floor plane and the origin grid under the stage ROOT — the SCENE
// frame, volume.json's; the frustums, the camera manipulator, the crop
// markers and the ruler under the FILE group — raw .ply coordinates,
// cameras.json's and manual_views.json's. No overlay transforms its
// coordinates itself. One module per concept; the shared
// TransformControls scaffolding is manipulator.ts.

export { COL, cssHex, frameFromUpAxis, type VolumeFrame } from "./common";
export { FrustumSet } from "./frustums";
export { Manipulator } from "./manipulator";
export { CameraGizmo, FocusGizmo, VolumeGizmo,
         type CameraMode, type FocusMode, type GizmoMode } from "./gizmos";
export { OriginGrid } from "./originGrid";
export { FloorPlane } from "./floorPlane";
export { CropMarkers } from "./cropMarkers";
export { Ruler, type Pick } from "./ruler";
export { type SplatArrays } from "./pick";
