// SPDX-FileCopyrightText: 2026 Dpt. <https://www.dpt.co>
// SPDX-License-Identifier: GPL-3.0-or-later

import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "node:path";

// The splat-canvas core is plain JS in viewer/lib/, outside this package:
// the aliases below resolve it, and its bare three/spark imports, to this
// package's pinned copies.
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: [
      { find: "@viewer",
        replacement: path.resolve(__dirname, "../viewer/lib") },
      // viewer/lib sits outside this root; point its bare specifiers at
      // our pinned copies, so the page holds one three and one spark.
      // Exact-match + addons mapping so three's package exports survive.
      { find: /^three$/,
        replacement: path.resolve(
          __dirname, "node_modules/three/build/three.module.js") },
      { find: /^three\/addons\//,
        replacement: path.resolve(
          __dirname, "node_modules/three/examples/jsm") + "/" },
      { find: /^@sparkjsdev\/spark$/,
        replacement: path.resolve(
          __dirname, "node_modules/@sparkjsdev/spark") },
    ],
    dedupe: ["three", "@sparkjsdev/spark"],
  },
  server: {
    fs: { allow: [path.resolve(__dirname, ".."), __dirname] },
    proxy: { "/api": "http://127.0.0.1:8090" },   // dev only
  },
  build: { outDir: "dist", chunkSizeWarningLimit: 4000 },
});
